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
        #os.system(f"python3 -m pip {' '.join(self.args)} > {fout} 2> {ferr}")
        os.system(f"ls -ltr /home_nfs/gsilvi/.local/lib/python3.9/site-packages/{' '.join(self.args)} > {fout} 2> {ferr}")
        #os.system(f"python3 -c 'import sys; print(sys.path)' > {fout} 2> {ferr}")
        #os.system(f"""python3 -c 'import os; import importlib;  print(os.path.abspath(importlib.__file__))' > {fout} 2> {ferr}""")
        #os.system(f"""python -c 'import os;import sys;sys.path.remove("/usr/lib64/python3.9");sys.path.append(os.path.expanduser("/home_nfs/gsilvi/.local/lib/python3.9/site-packages/"));import zipfile;  print(os.path.abspath(zipfile.__file__))' > {fout} 2> {ferr}""")
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