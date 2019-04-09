from ruamel import yaml
import csv
import zipfile

import logging

logger = logging.getLogger('optimizer')


def read_yaml(yaml_file):
    global logger
    with open(yaml_file, 'r') as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except (FileNotFoundError, IOError, yaml.YAMLError) as e:
            logger.error(e)
        else:
            return yaml_data


def write_yaml(yaml_file, data):
    global logger
    with open(yaml_file, 'w') as stream:
        try:
            yaml.dump(data, stream, default_flow_style=False)
        except (IOError, yaml.YAMLError) as e:
            logger.error(e)


def persist_data(filename, data, mode):
    global logger
    try:
        with open(filename, mode) as stream:
            wr = csv.writer(stream, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerow(data)
    except (FileNotFoundError, IOError) as e:
        logger.error(e)


def zip_files(files, zip_filename):
    global logger
    try:
        zipf = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)
    except IOError as e:
        logger.error(e)
    else:
        try:
            for file in files:
                zipf.write(file)
        except (FileNotFoundError, IOError) as e:
            logger.error(e)
        finally:
            zipf.close()






