from qat.qlmaas.upload import MetaLocalPlugin
# package intalled: qiskit==0.30.0 numpy myqlm-interop[qiskit_binder] qiskit_nature==0.2.0 sympy==1.8 psutil



#install --user 
class PackageInstaller(metaclass=MetaLocalPlugin):
    """
    A remote plugin that uploads code to the remote QLM.

    """

    def __init__(self, args):
        print(f"Remotely executing: python3 -m pip {' '.join(args)}")
        self.args = args

    def install_packages(self):
        """
        Install the packages via pip
        """
        import os
        fout = os.path.expanduser("~/pip_log_out")
        ferr = os.path.expanduser("~/pip_log_err")
        os.system(f"python3 -m pip {' '.join(self.args)} > {fout} 2> {ferr}")
        #install without looking into main folder for libraries
        #os.system(f"""export PYTHONPATH="/usr/lib/python3.9/site-packages:/home_nfs/gsilvi/.local/lib/python3.9/site-packages/" ; python3 -m pip {' '.join(self.args)} > {fout} 2> {ferr}""")
        #os.system(f"""export PYTHONPATH=$PYTHONPATH+":/home_nfs/gsilvi/.local/lib/python3.9/site-packages/" ;echo $PYTHONPATH > {fout} 2> {ferr}""")
        #os.system(f'echo "__all__ = [\"interop\", \"core\", \"plugins\",\"qlmaas\",\"qpus\"]" >> /home_nfs/gsilvi/.local/lib/python3.9/site-packages/qat/__init__.py > {fout} 2> {ferr}')
        #os.system(f"""ls ~/custom_qlm_code/qiskit_mod/qiskit_ter> {fout} 2> {ferr}""")
        #os.system(f"""python3 -c 'import os; import qat;  print(os.path.abspath(qat.__file__))' > {fout} 2> {ferr}""")
        #os.system(f"""python3 -c 'import os;import sys;sys.path.remove("/usr/local/lib64/python3.9/site-packages");sys.path.remove("/usr/local/lib/python3.9/site-packages");sys.path.remove("/usr/lib64/python3.9/site-packages");sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/lib/python3.9/site-packages/"));from importlib import import_module; print(import_module("qat.interop"))' > {fout} 2> {ferr}""")
        #os.system(f"""python3 -c 'import os;import sys;import importlib.util; spec = importlib.util.spec_from_file_location("qat.interop.qiskit","/home_nfs/gsilvi/.local/lib/python3.9/site-packages/qat/interop/qiskit/__init__.py",submodule_search_locations=['']);module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module); print(module, module.__file__); print(module.qiskit_to_qlm)' > {fout} 2> {ferr}""")
        #### qiskit version below
        # os.system(f"""python3 -c 'import os;import sys;\
        #             sys.path.remove("/usr/local/lib64/python3.9/site-packages");\
        #             sys.path.remove("/usr/local/lib/python3.9/site-packages");\
        #             sys.path.remove("/usr/lib64/python3.9/site-packages");\
        #             sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/lib/python3.9/site-packages/"));\
        #             import qat;\
        #             print(os.path.abspath(qat.__file__))' > {fout} 2> {ferr}""")
        #os.system(f"""python3 -c 'import sys; import os;sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/lib/python3.9/site-packages/")); import qiskit;  print(qiskit.__qiskit_version__)' > {fout} 2> {ferr}""")
        #os.system(f"cp /usr/lib64/python3.9/zipfile.py /home_nfs/gsilvi/.local/lib/python3.9/site-packages/zipfile.py > {fout} 2> {ferr}")
        data_out = ""
        data_err = ""
        with open(fout, 'r') as fin:
            data_out = fin.read()

        with open(ferr, 'r') as fin:
            data_err = fin.read()
        return data_out, data_err

    def compile(self, _batch, _specs):
        """
        """
        out, err = self.install_packages()
        _batch.meta_data = {"out": out, "err": err}
        return _batch

    def install_remotely(self):
        """
        Install the packages remotely
        """
        from qlmaas.plugins import UploadedPlugin
        from qat.core import Batch, HardwareSpecs
        plugin = UploadedPlugin(plugin=self)

        result = plugin.compile(Batch(), HardwareSpecs())
        result = result.join()
        print(result.meta_data['out'])
        print(result.meta_data['err'])


if __name__ == "__main__":
    import sys
    installer = PackageInstaller(sys.argv[1:])
    installer.install_packages()
    installer.install_remotely()