import os
import re
import csv
import cellpy
import shutil
import pandas as pd
from galvani import BioLogic

from batteryabn import logger, Constants as Const
from batteryabn.utils import Utils

def create_parser():
    return Parser()

class Parser:
    def __init__(self):
        """
        A class to parse data from battery test data files.
        """
        self.test_name = None
        self.test_type = None
        self.test_size = None
        self.raw_test_data = pd.DataFrame(dtype=object)
        self.raw_metadata = {}
        self.calibration_parameters = {}
        # Define parse functions
        self.parse_functions = {
            Const.ARBIN: self.parse_arbin,
            Const.BIOLOGIC: self.parse_biologic,
            Const.NEWARE: self.parse_neware,
            Const.VDF: self.parse_vdf,
        }

    def parse(self, file_path: str) -> None:
        """
        Parse data from battery test data files.
        Supported test types: 'BioLogic', 'Neware', 'Vdf'
        The data will be stored in the attributes of the class.

        Parameters
        ----------
        file_path : str
            Path to battery test data file
        """
        logger.debug(f'Load file path: {file_path}')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Unable to load file {file_path}')
        
        # Clear previous data before parsing new data
        self.clear()
        
        # Get measurement name
        self.test_name = self.__get_test_name(file_path)

        # Determine test type
        self.test_type = self.__determine_test_type(file_path)

        # Get the size of the test data
        self.test_size = self.__get_test_size(file_path)
        
        # Parse metadata from test name
        self.parse_metadata(self.test_name, self.test_type)

        # Parse data based on test type
        self.parse_functions[self.test_type](file_path)

    def parse_arbin(self, file_path: str) -> None:
        """
        Parse Arbin test data.

        Parameters
        ----------
        file_path : str
            Path to Arbin test data file
        """
        # Create a backup of the file before reading it, the modified cellpy read function will change the file
        # backup_file_path = Utils.backup_file(file_path)
        arbin_raw_df, arbin_summary_df, arbin_steps_df = self.__read_cellpy(file_path)

        # Store raw test data
        self.raw_test_data = arbin_raw_df
        # Utils.restore_file(backup_file_path)

    def parse_biologic(self, file_path: str) -> None:
        """
        Parse BioLogic test data.

        Parameters
        ----------
        file_path : str
            Path to BioLogic test data file
        """
 
        biologic_raw_df, start_time = self.__load_mpr(file_path)
        if biologic_raw_df is None:
            return
        # Add the timestamp column
        total_time = pd.to_timedelta(biologic_raw_df['time/s'])
        biologic_raw_df[Const.TIMESTAMP] = start_time + total_time

        self.raw_test_data = biologic_raw_df
        

    def parse_neware(self, file_path: str) -> None:
        """
        Parse Neware test data.

        Parameters
        ----------
        file_path : str
            Path to Neware test data file
        """
        neware_raw_df = self.__load_xlsx(file_path)

        # Add the timestamp for the neware data
        start_time = pd.to_datetime(neware_raw_df['Date'][0])
        total_time = pd.to_timedelta(neware_raw_df['Total Time'])
        # Adjust Total Time
        first_total_time = total_time[0]
        total_time = total_time - first_total_time
        neware_raw_df[Const.TIMESTAMP] = start_time + total_time
        neware_raw_df.drop(columns=['t1(℃)'], inplace=True)
        
        # Store raw test data
        self.raw_test_data = neware_raw_df

    def parse_vdf(self, file_path: str) -> None:
        """
        Parse Neware Vdf test data.

        Parameters
        ----------
        file_path : str
            Path to Neware Vdf test data file
        """
        vdf_df, vdf_meta = self.__load_vdf_csv(file_path)

        if vdf_df is None or vdf_meta is None:
            return
        # Store raw test data
        self.raw_test_data = vdf_df
        # Store raw metadata
        self.raw_metadata.update(vdf_meta)

    def parse_metadata(self, test_name: str, test_type: str) -> None:
        """
        Parse metadata from battery test name and type.
        Supported test types: 'Arbin', 'BioLogic', 'Neware', 'Vdf'
        The name of the test should follow the name rules of the test type.

        Parameters
        ----------
        test_name : str
            Name of the battery test
        test_type : str
            Type of battery test. Supported types: 'Arbin', 'BioLogic', 'Neware', 'Vdf'

        Raises
        ------
        ValueError
            If the test name does not match the name rules of the test type
        """
        keys = getattr(Const, f'{test_type.upper()}_NAME_KEYS')
        # Split the test name by '_' and match the keys
        try:
            values = test_name.split('_')
            self.raw_metadata.update(dict(zip(keys, values)))
        except:
            raise ValueError(f'Test name {test_name} does not match the name rules of {test_type}')
        

    def parse_calibration_parameters(self, file_path: str) -> None:
        """
        Get the calibration parameters from the csv file.
        TODO: The whole calibration in db or better config file.
        
        Parameters
        ----------
        file_path : str
            The path to the calibration csv file
        """
        calibration_parameters = {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            logger.info(f"Loaded csv file from {file_path} successfully")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return calibration_parameters

        calibration_csv = iter(rows)
        header = next(calibration_csv)

        #TODO: The format of the calibration file is not good, need to be updated
        project_index = header.index('Project')
        cell_name_index = header.index('Cell Name')
        x1_index = header.index('X1')
        x2_index = header.index('X2')
        c_index = header.index('C')
        start_date_index = header.index('Start Date (Aging)')
        removal_date_index = header.index('Removal Date')
        protocol_index = header.index('Test Protocol')

        # TODO: Default values should be fetched from the database in the future
        default_X1, default_X2, default_C = 1.6473, -27.134, 138.74

        for row in calibration_csv:
            project = row[project_index]
            cell_number = row[cell_name_index]
            start_date = row[start_date_index] if row[start_date_index] else None
            removal_date = row[removal_date_index] if row[removal_date_index] else '01/01/2050'

            x1 = float(row[x1_index]) if row[x1_index] else default_X1
            x2 = float(row[x2_index]) if row[x2_index] else default_X2
            c = float(row[c_index]) if row[c_index] else default_C

            cell_name = f"{project}_CELL{cell_number.zfill(3)}"

            if cell_name not in calibration_parameters:
                calibration_parameters[cell_name] = []

            protocol = row[protocol_index] if row[protocol_index] else 'DEFAULT'
            # calibration_parameters[cell_name].append({protocol: (x1, x2, c)})
            calibration_parameters[cell_name].append({protocol:(start_date, removal_date, x1, x2, c)})
        self.calibration_parameters = calibration_parameters
        
    def __get_test_name(self, file_path: str) -> str:
        """
        Get measurement name from file path.

        Parameters
        ----------
        file_path : str
            Path to battery test data file

        Returns
        -------
        str
            Measurement name
        """
        logger.debug(f'Get measurement name from file path: {file_path}')
        measurement_name = None

        # Get measurement name from file path
        file_name = os.path.basename(file_path)
        measurement_name = re.sub(r'\..*$', '', file_name)
        logger.debug(f'Measurement name: {measurement_name}')

        return measurement_name

    def __determine_test_type(self, file_path: str) -> str:
        """
        Determine test type from file path.

        Parameters
        ----------
        file_path : str
            Path to battery test data file

        Returns
        -------
        str
            Test type
        """
        logger.debug(f'Determine test type from file path: {file_path}')
        test_type = None

        # Determine test type from file path
        file_type = file_path.split(".")[-1]

        if file_type in Const.FILE_TYPE_2_TEST_TYPE:
            test_type = Const.FILE_TYPE_2_TEST_TYPE[file_type]
            logger.debug(f'Test type: {test_type}')
        else:
            raise ValueError(f'Unsupported file type: {file_type}')
        
        return test_type
    
    def __get_test_size(self, file_path: str) -> int:
        """
        Get the size of the test data.

        Parameters
        ----------
        file_path : str
            Path to the test data file

        Returns
        -------
        int
            The size of the test data
        """
        try:
            # Get the size of the test data
            test_size = os.path.getsize(file_path)
            logger.info(f"Test data size: {test_size} bytes")
            return test_size
        except:
            raise ValueError(f"Failed to get the size of the test data from {file_path}")

    def __load_xlsx(self, file_path: str, sheet: str = 'record') -> pd.DataFrame:
        """
        Read the xlsx file, get the sheet and return the raw data

        Parameters
        ----------
        file_path : str
            The path to the xlsx file

        sheet : str (default: 'record')
            The sheet name to read 
            
        Returns
        -------
        raw_data : pandas.DataFrame
            The raw data
        """
        try:
            xlsx = pd.ExcelFile(file_path, engine='openpyxl')
            # Get the 'record' sheet
            raw_data = pd.read_excel(xlsx, sheet_name=sheet)
            logger.info(f"Loaded xlsx file from {file_path} successfully")
            
            return raw_data

        except:
            raise ValueError(f"Failed to load xlsx file from {file_path}")
        
    def __load_vdf_csv(self, file_path: str) -> pd.DataFrame:
        """
        Read the vdf csv file and return the raw data

        Parameters
        ----------
        file_path : str
            The path to the vdf csv file
            
        Returns
        -------
        vdf_df : pandas.DataFrame
            The raw data of the vdf csv file

        metadata : dict
            The metadata inside the vdf csv file
        """
        vdf_meta = {}
        start_line = None

        try:
            # Open the file once and parse it line by line
            with open(file_path, 'r') as file:
                for i, line in enumerate(file):
                    if "[DATA START]" in line:
                        start_line = i + 1  # Adjust for 0-based index
                        break  # Exit loop once the data start marker is found
                    elif ':' in line:
                        key, value = line.split(':', 1)  # Split on the first colon only
                        vdf_meta[key.strip()] = value.strip()
        except:
            raise ValueError(f"Failed to read vdf meta data from {file_path}")

        # Check if the data start marker was not found
        if start_line is None:
            logger.error(f"Data start marker not found in {file_path}")
            return None, None

        try:
        # Read the DataFrame, assuming the first row of data contains units
            df_units = pd.read_csv(file_path, delimiter='\t', skiprows=start_line, nrows=1)
            header = df_units.columns
            units = df_units.iloc[0].values

            # New header with units if available
            new_header = [f"{header[i]} ({units[i]})" if units[i] != 'none' else header[i] for i in range(len(header))]

            # Read the actual data, skipping the units row
            vdf_df = pd.read_csv(file_path, delimiter='\t', skiprows=start_line + 2, header=None, names=new_header)

            logger.info(f"Loaded vdf csv file from {file_path} successfully")
        except Exception as e:
            logger.error(f"Failed to read vdf data from {file_path}: {e}")
            return None, None
        
        return vdf_df, vdf_meta

    def __load_mpr(self, file_path: str) -> pd.DataFrame:
        """
        Read the mpr file of Biologic, get the data and return the dataframe

        Parameters
        ----------
        file_path : str
            The path to the mpr file

        Returns
        -------
        df : pandas.DataFrame
            The dataframe
        start_timestamp : datetime.datetime
            The start timestamp
        """
        try:
            # Read the mpr file
            mpr_file = BioLogic.MPRfile(file_path)
            # Get the start time
            # If the mpr_file.timestamp is not available, use the startdate
            if hasattr(mpr_file, 'timestamp') and mpr_file.timestamp:
                start_time = mpr_file.timestamp
            else:
                start_date = mpr_file.startdate
                # Start date is datetime.date type, convert it to datetime.datetime
                start_time = pd.to_datetime(start_date)

            df = pd.DataFrame(mpr_file.data)
            logger.info(f"Loaded mpr file from {file_path} successfully")

            return df, start_time
        
        except Exception as e:
            logger.error(f"Failed to load mpr file from {file_path}: {e}")
            return None, None
        
    def __read_cellpy(self, file_path: str):
        """
        Read the cellpy file and return the iterator of rows

        Parameters
        ----------
        file_path : str
            The path to the cellpy file

        Returns
        -------
        raw_data : pandas.DataFrame
            The raw data
        summary_data : pandas.DataFrame
            The summary data
        steps_data : pandas.DataFrame
            The steps data
        """
        try:
            # Use cellpy to read the file and convert it to a list of rows
            temp_dir = "/home/me-bcl/BatteryABN/tmp/cellpy_temp"
            # Create a temporary directory if it does not exist
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))
            cellpy.prms.Instruments.Arbin['max_res_filesize'] = 1000000000
            # Copy the file to the temp directory
            shutil.copy(file_path, temp_file_path)
            cell = cellpy.cellreader.get(temp_file_path, instrument="arbin_res")
            data = cell.data
            raw_data = data.raw
            summary_data = data.summary
            steps_data = data.steps
            logger.info(f"Loaded cellpy file from {file_path} successfully")
            # Delete the temp file
            os.remove(temp_file_path)

            return raw_data, summary_data, steps_data
            
        except Exception as e:
            logger.error(f"Failed to load cellpy file from {file_path}: {e}")
            return None, None, None

    def clear(self) -> None:
        """
        Clear the previous data.
        """
        self.test_name = None
        self.test_type = None
        self.raw_test_data = pd.DataFrame(dtype=object)
        self.raw_metadata = {}