from qat.qlmaas.upload import MetaLocalPlugin


class Uploader(metaclass=MetaLocalPlugin):
    """
    A remote plugin that uploads code to the remote QLM.

    """

    def __init__(self, path="~/",localpath="~/",  files=None): 
        self.files = files or []
        self.path = path
        self.localpath = localpath
        self.files_content = []
        self.load_files()

    def load_files(self):
        """
        Read the files and store their content.
        """
        for fname in self.files: #  Read the files
            fname = os.path.expanduser(os.path.join(self.localpath, fname)) 
            print("%" * 20, f"{fname}", "%" * 20) #  Print the name of the file
            with open(fname, "r") as fin: #  Read the file
                content = fin.read() #  Store the content
                print(content) #  Print the content
                self.files_content.append(content) #  Store the content
            print("%" * 50)

    def deploy_files(self):
        """
        Deploys the stored files in the given path.
        """
        import os
        os.system(f"mkdir -p {self.path}") #  Create the directory
        for fname, fcontent in zip(self.files, self.files_content): #  For each file and its content 
            fname = os.path.expanduser(os.path.join(self.path, fname))  #  Expand the path
            print('Writing file:', fname)
            with open(fname, "w") as fout: #  Write the content to the file
                fout.write(fcontent) #  Write the content

    def compile(self, _batch, _specs):
        """
        Triggers the copy of the files in the correct spot.
        """
        self.deploy_files() #  Copy the files to the remote server
        return _batch
    
    def upload(self):
        """
        Upload the files to the remote server.
        """
        from qat.qlmaas import QLMaaSConnection
        from qlmaas.plugins import UploadedPlugin
        from qat.core import Batch, HardwareSpecs
        plugin = UploadedPlugin(plugin=self) #  Create the plugin
        print("Starting upload and updating the remote configuration")
        result = plugin.compile(Batch(), HardwareSpecs()) #  Compile the plugin, this will trigger the copy of the files
        print("Updating remote configuration (to publish the code)...")
        # connection = QLMaaSConnection() #  Create the connection
        # new_paths = [self.path] #  Create the new paths
        # new_modules = [] #  Create the new modules
        # for fname in self.files: #  For each file
        #     module_name = fname.replace(".py", "").replace("./", "") #  Get the module name, removing the extension
        #     new_modules.append(module_name) #  Add the module name to the list
        # print(">> uploading new configuration")
        # connection.update_config(new_paths, new_modules, False) #  Update the configuration, this will publish the code
        # print("Checking if the upload finished")
        # result.join() #  Wait for the upload to finish
        # print("Done!")
        # config = connection.get_config() #  Get the configuration
        # new_paths = list(set(config.paths)) #  Get the new paths
        # new_modules = list(set(config.modules)) #  Get the new modules  
        # connection.update_config(new_paths, new_modules, True) #  Update the configuration, this will publish the code
        # import os
        # # os.system("python3 -m pip list")
        # # os.system("python3 --version")
        # print(f"The new configuration is:\n{connection.get_config()}")

import os
import itertools
path1 = "./qiskit_mod/qiskit_nat/"
path2 = "./qiskit_mod/qiskit_ter/"

paths = [path1, path2]
# for (dirpath, dirnames, filenames) in itertools.chain.from_iterable(os.walk(path) for path in paths):
#     print(dirpath)
#     filenames = [dirpath + s for s in filenames]
#     f.extend(filenames)
for path in paths:
    f = []
    for item in os.listdir(path):
        path_to_file = os.path.join(path, item)
        if os.path.isfile(path_to_file):
            print(path_to_file)
            f.append(item)

    print(f)
    uploader = Uploader(path="~/custom_qlm_code/"+path, localpath=path, files=f)
    uploader.deploy_files()
    uploader.upload()
