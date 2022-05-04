from qat.qlmaas.upload import MetaLocalPlugin


class Uploader(metaclass=MetaLocalPlugin):
    """
    A remote plugin that uploads code to the remote QLM.

    """

    def __init__(self, path="~/", files=None):
        self.files = files or []
        self.path = path
        self.files_content = []
        self.load_files()

    def load_files(self):
        """
        Read the files and store their content.
        """
        for fname in self.files:
            print("%" * 20, f"{fname}", "%" * 20)
            with open(fname, "r") as fin:
                content = fin.read()
                print(content)
                self.files_content.append(content)
            print("%" * 50)

    def deploy_files(self):
        """
        Deploys the stored files in the given path.
        """
        import os
        os.system(f"mkdir -p {self.path}")
        for fname, fcontent in zip(self.files, self.files_content):
            fname = os.path.expanduser(os.path.join(self.path, fname))
            with open(fname, "w") as fout:
                fout.write(fcontent)

    def compile(self, _batch, _specs):
        """
        Triggers the copy of the files in the correct spot.
        """
        self.deploy_files()
        return _batch

    def upload(self):
        """
        Upload the files to the remote server.
        """
        from qat.qlmaas import QLMaaSConnection
        from qlmaas.plugins import UploadedPlugin
        from qat.core import Batch, HardwareSpecs
        plugin = UploadedPlugin(plugin=self)
        print("Starting upload and updating the remote configuration")
        result = plugin.compile(Batch(), HardwareSpecs())
        print("Updating remote configuration (to publish the code)...")
        connection = QLMaaSConnection()
        new_paths = [self.path]
        new_modules = []
        for fname in self.files:
            module_name = fname.replace(".py", "").replace("./", "")
            new_modules.append(module_name)
        print(">> uploading new configuration")
        connection.update_config(new_paths, new_modules, False) 
        print("Checking if the upload finished")
        result.join()
        print("Done!")
        config = connection.get_config()
        new_paths = list(set(config.paths))
        new_modules = list(set(config.modules))
        connection.update_config(new_paths, new_modules, True)
        import os
        # os.system("python3 -m pip list")
        # os.system("python3 --version")
        print(f"The new configuration is:\n{connection.get_config()}")


uploader = Uploader(path="~/custom_qlm_code/", files=["./my_junction.py"])
uploader.deploy_files()
uploader.upload()
