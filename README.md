# NEASQC repo Variatonal Algorithms

This repositorycollects Python scripts and Jupyter notebooks that allow the user to test different variational algorithms. It contains our custom functions (e.g. VHA ansatz, PBO Hamiltonian)
that are built upon Qiskit libraries.

The repository is organized as follows:
- **misc**  contain the notebooks and scripts that showcase the variational algorithms
- **qiskit_mod** contains our custom functions that are built upon Qiskit libraries as well as the QLM custom junction and helpers:
    - **qiskit_nat** the customize function built upon qiskit_nature
    - **qiskit_ter** the customize function built upon qiskit_terra
    - *my_junction.py* the QLM custom junction used to run the variational algorithms with QLM backends
    - *uploader_junction.py* helper to upload the junction to QLMaaS server, so that it can be found in the remote library
    - *wrapper2myqlm.py* helper to wrap the variational algorithms using QLM stack
- **tests** unit tests for the variational algorithms
- **QLMtools** additional tools to upload qiskit_mod to QLMaaS server
    - *uploader_library.py* helper to upload the qiskit_mod library to QLMaaS server
## Licence

The `LICENCE` file contains the default licence statement as specified in the proposal and partner agreement.

## Building and installing

To run the code in the repo a setup to build the conda environment is provided. 
It install python 3.9, qiksit libraries, and the two mods to the following library: qiskit-nature and qiskit-terra.
These two repos are modified to include additional functionalities not present in the standard qiskit libraries.
Additionaly the conda enviroment install QLM libraries necessary to use QLM QPUs as backend.

To install the conda environment run the following command:
```bash
source create_conda_env.sh
```

Also, keep in mind that recently Github password authentication has been deprecated and will no longer work.
Instead, token-based authentication (for example SSH Key) are required for all authenticated Git operations.

## Running the code
You can find Jupyter notebook and python scripts in misc folder.
Use the conda environment to run the code.


## QLM interoperatibility explained
The code in the repository is mainly written using the Qiskit library. To be able to run the circuits onto QLM quantum processing units (QPUs), we integrated the myqlm-interop library which enables the conversion of Qiskit circuits to QLM circuits (as well as the opposite).
Additionally, the library allows wrapping QLM QPUs onto a Qiskit`s quantum instance. This allows for easy and simple integration of QPUs as backends to run the circuits. 
This feature wraps each circuit and observable into a QLM job that is submitted to either the local or remote QLM QPU.
Unfortunately, this implementation suffers from a big overhead due to the time associated with job submissions, result retrieval, and, possibly, queue waiting times.

For this reason, to minimize the overhead, we decided to use the custom plugin framework of MyQLM. In particular, we build a custom junction that is capable of handling multiple circuits runs onto a single job submission, as well as all the classical computations associated.
The same custom junction is uploaded to the QLM server to be available from the QLMaaS library when the remote connection is established.

The custom junction gets the various methods tested in the repository as input. All methods require multiple jobs to run for various circuits and the junction framework can handle them within a single submission to minimize overhead time. 
The custom junction also modifies the function to get the energy evaluation (and gradient) inside the Qiskit solver. 
The modification converts each circuit from Qiskit to MyQLM and takes care of the job submission in the QLM framework.

Overall, this Qiskit-QLM integration allows us to choose which type of backend to use, and when combined with the QLMaaS server, enables this code to run for larger problems and molecules, which would not be possible using a simple laptop.

## Testing and continuous integration

You can run the tests with:

```bash
python setup.py test
```


## Coding conventions

In order to simplify the coding conventions, we provide a pylint.rc file in `misc/pylint.rc`.
This will allow you to easily check your naming conventions and various other aspects.
This is not a strict guidline, as pylint can be quite pedantic sometimes (but also very helpful).

A few remarks:
- pylint can be integrated in most editors (and we strongly advise you to)
- running pylint on several source files in one go can find errors such as circular imports or code duplication:

```bash
python -m pylint --rcfile=./misc/pylint.rc <my_source_dir>
```
or

```bash
pylint --rcfile=./misc/pylint.rc <my_source_dir>
```

depending on how you installed pylint.




## GitHub CI
This repository contains a GitHub Workflow file that will automatically run pytest when changes are pushed.  
Details on disabling and enabling this feature can be found [here](https://docs.github.com/en/enterprise-server@3.0/actions/managing-workflow-runs/disabling-and-enabling-a-workflow).

## Documentation
Customize the GitHub Workflow YAML file: *repo_name*/.github/workflow/sphinx\_doc.yml
There is a basic index.rst template under 'doc'. Modify it as you want.

Any modification under the folder 'doc' pushed to GitHub will trigger a rebuild of the documentation (using GitHub CI).
If the build is successful, then the resulting html documentation can be access at: https://neasqc.github.io/repo_name

Notes:
  - You can follow the build process on the 'Actions' tab below the name of the repository.
  - neasqc.github.io does not immediately update. You may need to wait a minute or two before it see the changes.

