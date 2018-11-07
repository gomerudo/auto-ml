
################################################################################
######################### This is just a python script #########################
################################################################################
import requests
import os.path
import os
import tarfile
import re


################################################################################
############################## DECLARE CONSTANTS  ##############################
################################################################################
TmpStorageDir = "python-pkgs"
PyPiEndpoint = "https://pypi.org/pypi"
Format ="json"
MainLibraries = ["TPOT@0.9.5", "auto-sklearn@0.4.0", "openml@0.7.0"]

################################################################################
################################# SOME METHODS #################################
################################################################################

# Function to get the URL query endpoint for a given python store ###
def getLibraryQuery(library = None, version = None, endpoint = PyPiEndpoint, 
                                                            format = Format) :
    if library is None or version is None :
      raise ValueError('Library and version must be not null')
    
    if version == "latest" :
        return "{endpoint}/{library}/{format}".format(endpoint = endpoint,
                    library = library, format = format)
    
    return "{endpoint}/{library}/{version}/{format}".format(endpoint = endpoint,
                    library = library, version = version, format = format)

# Function to execute a REST api call and retrieve the response as a JSON
def executeQuery(restURL = None) :
    response = requests.get(restURL)
    try :
        data = response.json()
        return data['urls']
    except ValueError :
        print("Error while evaluating url {url}".format(url = restURL))
        return None
    
# Function to perform the search library on python store operation
def searchLibrary(library = None, version = None, 
                                        repositoryEndpoint = PyPiEndpoint) : 
    getURL = getLibraryQuery(library, version, PyPiEndpoint, Format)
    responseJson = executeQuery(getURL)    
    
    return responseJson
        
# This function retrieves the url from the json output of a python store
def getDownloadUrl(jsonOutput = None) :
    firstDownloadUrl = jsonOutput[0]['url']
    return firstDownloadUrl

# Function to download a file from a given URL into a given path
def downloadFile(url = None, fileName = None, directory = None) :
    target = "{dirname}/{basename}".format(dirname = directory, 
            basename = fileName)
    
    if not os.path.isdir(directory) :
        os.makedirs(directory)

    r = requests.get(url)
    
    if not os.path.isfile(target) :
    
        print('Downloading file "{fileName}" as {newName}'.format( 
            fileName = fileName, newName = target ) )
           
        with open(target, 'wb') as f:  
            f.write(r.content)
        return r.status_code
        
        # Retrieve HTTP meta-data
    return 0

def untar(filePath, directory):
    if filePath.endswith("tar.gz") :
    
        # Build the extracted directory/file path
        untardir = directory + "/" + os.path.basename(filePath)
        # This removes the .gz
        untardir = os.path.splitext(untardir)[0]
        # This removes the .tar
        untardir = os.path.splitext(untardir)[0]
        
        if not os.path.isdir(untardir) and not os.path.isfile(untardir):
            tar = tarfile.open(filePath)
            tar.extractall(directory)
            tar.close()

        return untardir
    else:
        return None

def readDependenciesFromFile(filePath, list, visited = []) :
    with open(filePath) as f:
        content = f.readlines()
    
    for line in content :
        regexA = re.compile('^([A-Za-z]+)([><=][=])(.+)')
        matchA = re.match(regexA, line.strip()) 
        
        regexB = re.compile('^([A-Za-z]+)')
        matchB = re.match(regexB, line.strip()) 
        element = "{name}@{version}"
        # Case A: name==version
        if matchA :
            name = matchA.group(1)
            constraint = matchA.group(2)
            version = matchA.group(3)
            if constraint == ">=" : version = "latest"
            element = element.format(name = name, version = version) 
        # Case B: name
        elif matchB :
            name = matchB.group(1)
            element = element.format(name = name, version = "latest") 
        else :
            print("Could not find pattern for {line}".format(line = line.strip()))
        
        if element and element not in list and element not in visited :
                print("Appending {} ".format(element))
                list.append(element)
    return list
                

    
def addDependenciesToList(list = [], pkgDir = None, visited = []) : 
    pkgName = os.path.basename(pkgDir)
    
    aux = pkgName.split("-")
    pkgVersion = aux[len(aux) - 1]
    
    pkgName = pkgName.replace("-" + pkgVersion, '')
    pkgName = pkgName.replace("-", "_")

    eggDir = "{pkgDir}/{pkgName}.egg-info".format(pkgDir = pkgDir, 
                pkgName = pkgName)
    depsFile = "{parent}/requires.txt".format(parent = eggDir)
    
    # Check if directory exists
    if os.path.isdir(eggDir) :
        if os.path.isfile(depsFile) :
            readDependenciesFromFile(depsFile, list, visited)
        else : 
            print("{dir} directory exists but no {req} file is there".format( 
                dir = eggDir, req = "requires.txt" ))
    else : 
        print("{dir} does not exist".format(dir = eggDir))
    
    return list
    
################################################################################
##################################### MAIN #####################################
################################################################################

def main() : 
    workspace = "pyworkspace/libraries"
    auxList = MainLibraries.copy()
    visited = []
    while auxList :
        library = auxList.pop()
        nameVersion = library.split("@")
        libraryName = nameVersion[0]
        libraryVersion = nameVersion[1]
        
        # Download the file
        jsonOutput = searchLibrary(libraryName, libraryVersion, PyPiEndpoint)
        if jsonOutput : 
            downloadUrl = getDownloadUrl(jsonOutput)
            # TODO: verify checksums
            fileBaseName = os.path.basename(downloadUrl)
            downloadFile(downloadUrl, fileBaseName, workspace)
            
            print()
            # Untar the file
            dest = untar("{workspace}/{filename}".format(workspace = workspace, 
                filename = fileBaseName), workspace)
            if not dest :
                print("Could not perform any operation on {}".format(fileBaseName))
            else :
                print("Retrieving dependencies in {pkgDir}".format(pkgDir = dest))
                addDependenciesToList(auxList, dest, visited)
            
            visited.append(library)
        # Read and add dependencies to the MainLibraries
        

main()

