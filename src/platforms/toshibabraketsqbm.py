"""
Copyright (c) 2023 Objectivity Ltd.
"""

import json
import logging
import os
import tempfile
from typing import Any, Union, cast
import urllib.request as ur
import urllib.parse as up
import numpy as np
from qiskit_optimization import QuadraticProgram  # type: ignore
import scipy.sparse as ss  # type: ignore
import scipy.io as sio  # type: ignore
import h5py  # type: ignore

from models.model import ModelStep
from models.qubo import ToQuboNumpyCallback

from platforms.awsbraket import AWSBraket


class ToshibaBraketSQBM(AWSBraket):
    """
    The Platform class for the Braket-hosted Toshiba SQBM+ optimisation service.
    It supports the following flags in its configuration section:

      - `url` the Toshiba solver url, e.g. https://sqbmplus-nbd.net
      - `token` the authentication token (if applicable)
      - `chunked` whether to use chunked encoding to transfer the model, default `False`
      - `format` what file format to use for the model, `mm` or `hf5` (default)
      - `solver` a subsection with solver parameters:
      - `type` one of `ising` or `autoising`, or 'qubo' for the newer versions of
      - any further parameters taken by the solver
    """

    def __init__(self, config: dict) -> None:
        """
        :param config: the model section of the configuration file
        """
        super().__init__(config)

        if "url" not in config:
            raise ValueError("Toshiba solver url not specified")
        self._url = config["url"]
        self._authorisation = (
            {"Authorization": f'Bearer {config["token"]}'} if "token" in config else {}
        )
        self._chunked = config.get("chunked", False)  # chunked content transfer of POST
        self._hf5 = config.get("format", "hf5") == "hf5"

        if "solver" not in config:
            raise ValueError("Toshiba solver configuration not specified")
        self._solver_config = config["solver"]

        resp = ur.urlopen(
            ur.Request(url=f"{self._url}/version", headers=self._authorisation)
        )
        if resp.status != 200:
            raise RuntimeError(f"Toshiba http response {resp.status} ({resp.reason})")
        logging.info("%s version %s", self, json.loads(resp.read().decode())["version"])

    def _solver_type(self) -> str:
        """
        :returns the solver type used, `"ising"` by defalt
        """
        return self._solver_config.get("type", "ising")

    def translate_problem(self, step: ModelStep) -> np.ndarray:
        """
        Translate the problem into a numpy matrix.

        :returns the interaction matrix
        """
        callback = ToQuboNumpyCallback()
        self.construct_qubo(step, callback)
        return callback.interactions

    def num_variables(self, problem: Any) -> int:
        """
        :returns the number of variabes in the problem.
        """
        interactions = cast(np.ndarray, problem)
        return len(interactions)

    def solve(self, problem: Any, timeout: int, num_solutions_desired: int) -> dict:
        interactions = cast(np.ndarray, problem)

        def is_sparse():
            # Determine if the interaction matrix is sparse enough
            nonzero_interactions = np.count_nonzero(interactions)
            logging.debug(
                "problem has %d/%d nonzero interactions",
                nonzero_interactions,
                interactions.size,
            )
            return nonzero_interactions / interactions.size < 0.3

        def hf_section(group, name, data, dtype):
            # Write a section in hf5 format
            temp = group.create_dataset(
                name=name,
                shape=data.shape,
                dtype=dtype,
                compression="gzip",
                compression_opts=1,
            )
            temp[:] = data[:]
            return temp

        def hf_write_problem(group):
            # Write the problem to the hf5 file
            if is_sparse():
                sparse = ss.csr_matrix(interactions)
                hf_section(group, "data", sparse.data, np.float32).attrs[
                    "format"
                ] = "csr"
                hf_section(group, "indices", sparse.indices, np.uint32)
                hf_section(group, "indptr", sparse.indptr, np.uint32)
            else:
                hf_section(group, "data", interactions.data, np.float32).attrs[
                    "format"
                ] = "dense"

        def submit_problem(file):
            # Submit the problem to the Toshiba server, streaming in from file
            parameters = (
                {k: v for k, v in self._solver_config.items() if k != "type"}
                | ({"timeout": timeout} if timeout else {})
                | (
                    {"maxout": num_solutions_desired}
                    if num_solutions_desired > 1
                    else {}
                )
            )
            uri = f"{self._url}/solver/{self._solver_type()}?{up.urlencode(parameters)}"

            headers = self._authorisation | {"Content-Type": "application/octet-stream"}
            if not self._chunked:
                file.seek(0, os.SEEK_END)
                headers |= {"Content-Length": file.tell()}
                file.seek(0, os.SEEK_SET)

            logging.debug("submitting problem to %s (%s)", uri, headers)
            resp = ur.urlopen(ur.Request(url=uri, headers=headers, data=file))

            content = resp.read()
            if content:
                response = json.loads(content)
                logging.debug(
                    "job id=%s time=%s wait=%s runs=%s message=%s",
                    response["id"],
                    response["time"],
                    response["wait"],
                    response["runs"],
                    response["message"],
                )

            if resp.status != 200:
                raise RuntimeError(
                    f"Toshiba http response {resp.status} ({resp.reason})"
                )

            return response

        if self._hf5:
            try:
                fname = None
                # Send the request in HF5 format
                with tempfile.NamedTemporaryFile(delete=False) as file_write:
                    fname = file_write.name
                    hf_write_problem(h5py.File(file_write, "w").create_group("/qubo"))
                with open(fname, "rb") as file_read:
                    return submit_problem(file_read)
            finally:
                if fname:
                    os.remove(fname)
        else:
            with tempfile.TemporaryFile() as file_temp:
                # Send the request in MatrixMarket format
                if is_sparse():
                    interactions = ss.coo_array(interactions)
                sio.mmwrite(file_temp, interactions, field="real", symmetry="symmetric")
                return submit_problem(file_temp)

    def get_info(self, result: Any) -> str:
        """
        Extracts the `autoising`-generated parameters from the result, if present.

        :param result: the Toshiba server response
        :returns the generated parameters or an empty string
        """
        response = cast(dict, result)
        params = response.get("param", response)
        return " ".join(
            map(
                lambda key: f"{key}={params[key]}",
                filter(lambda key: key in params, ["algo", "dt"]),
            )
        )

    def get_solver_time(self, result: Any) -> float:
        """
        Return the actual solver time used in seconds.

        :param result: the Toshiba server response
        :returns the solver time in seconds
        """
        response = cast(dict, result)
        return float(response["time"])

    def translate_result(
        self,
        step: ModelStep,
        qubo: QuadraticProgram,
        result: Any,
        num_solutions_desired: int,
    ) -> Union[list, np.ndarray]:
        """
        Translates the response `dict` into variable values for the `QuadraticProgram`.

        :param step: the `ModelStep` executing the optimisation problem
        :param qubo: the `QuadraticProgram` representing the model being solved
        :param result: the server response containing the optimised variable assignment
        :param num_solutions_desired: how many solutions should be returned
        :returns the optimised variable assignment(s) in terms of the `QuadraticProgram`
        """
        response = cast(dict, result)
        if num_solutions_desired > 1:
            results = [step.from_qubo(response["result"])]
            if "others" in response:
                results += [
                    step.from_qubo(other["result"]) for other in response["others"]
                ]
            return results

        else:
            return step.from_qubo(response["result"])

    def __str__(self) -> str:
        """
        :returns a string representation for logging purposes
        """
        return f"ToshibaBraketSQBM+-{self._solver_type()}"
