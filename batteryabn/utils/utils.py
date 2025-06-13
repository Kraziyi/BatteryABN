import os
import logging
import dotenv
import json
import yaml
import gzip
import pickle
import pandas as pd
import numpy as np
import io
import shutil
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from batteryabn import logger, Constants as Const


class Utils:

    @staticmethod
    def load_env(env_path: str) -> None:
        """
        Load environment variables from .env file

        Parameters
        ----------
        env_path : str
            Path to .env file
        """
        logger.info(f'Load environment variables from {env_path}')
        dotenv.load_dotenv(env_path, override=True)
        # Set logger level
        if os.getenv('ENV') == 'dev':
            logger.info('Set logger level to DEBUG')
            logger.setLevel(logging.DEBUG)  
        else:
            logger.info('Set logger level to INFO')
            logger.setLevel(logging.INFO)

    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        Load config file from path

        Parameters
        ----------
        config_path : str
            Path to config file

        Returns
        -------
        dict
            Config file as a dictionary
        """
        logger.info(f'Load config file from {config_path}')
        config = None

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file not found at {config_path}')

        # Try to load config file as yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info('Loaded config file as yaml')
                return config
        except yaml.YAMLError:
            logger.debug('Config file is not yaml')

        # Try to load config file as json
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info('Loaded config file as json')
                return config
        except json.decoder.JSONDecodeError:
            logger.debug('Config file is not json')

        # If config file is not yaml or json, raise an error
        raise ValueError('Config file is not yaml or json')
    
    @staticmethod
    def gzip_pikle_dump(data: pd.DataFrame, protocol=pickle.HIGHEST_PROTOCOL):
        """
        Dump data to a gzip compressed pickle bytes

        Parameters
        ----------
        data : pd.DataFrame
            Data to dump
        protocol : int, optional
            Pickle protocol, by default pickle.HIGHEST_PROTOCOL

        Returns
        -------
        bytes
            Compressed pickle data
        """
        logger.info(f'Dump data to gzip compressed pickle bytes')
        pickled_data = pickle.dumps(data, protocol=protocol)
        compressed_data = gzip.compress(pickled_data)
        
        return compressed_data
    
    @staticmethod
    def gzip_pickle_load(data):
        """
        Load data from a gzip compressed pickle bytes

        Parameters
        ----------
        data : bytes
            Compressed pickle data

        Returns
        -------
        pd.DataFrame
            Data loaded from compressed pickle data
        """
        logger.info(f'Load data from gzip compressed pickle bytes')
        pickled_data = gzip.decompress(data)
        data = pickle.loads(pickled_data)
        
        return data
    
    @staticmethod
    def formate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Format column names of a DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to format column names

        Returns
        -------
        pd.DataFrame
            DataFrame with formatted column names
        """
        logger.info(f'Format column names of DataFrame')
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.strip()
        return df

    @staticmethod
    def rename_columns(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
        """
        Rename columns of a DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to rename columns
        rename_dict : dict
            Dictionary with old column names as keys and new column names as values

        Returns
        -------
        pd.DataFrame
            DataFrame with renamed columns
        """
        logger.info(f'Rename columns of DataFrame')
        logger.debug(f"Columns name before renaming: {df.columns.tolist()}")

        # Normalize rename_dict keys to lower case and strip
        rename_dict = {k.lower().strip(): v for k, v in rename_dict.items()}

        df = df.rename(columns=rename_dict)
        logger.debug(f"Columns name after renaming: {df.columns.tolist()}")
        return df

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Drop columns from a DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to drop columns
        columns : list
            List of columns to drop

        Returns
        -------
        pd.DataFrame
            DataFrame with dropped columns
        """
        logger.info(f'Drop {len(columns)} columns from DataFrame')
        df = df.drop(columns=columns)
        return df

    @staticmethod
    def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        The function drops empty rows from DataFrame

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to drop empty rows from

        Returns
        -------
        df : pandas.DataFrame
            Pandas DataFrame with the empty rows dropped
        """
        all_na_count = (df.isna().all(axis=1)).sum()
        if all_na_count > 0:
            logger.info(f'Drop {all_na_count} empty rows')
            df = df.dropna(how='all')

        return df
    
    @staticmethod
    def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        The function drops unnamed columns from DataFrame

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to drop unnamed columns from.

        Returns
        -------
        df : pandas.DataFrame
            Pandas DataFrame with the unnamed columns dropped.
        """
        logger.info('Drop unnamed columns')
        unnamed_column_list = df.filter(like='Unnamed', axis=1).columns
        df = Utils.drop_columns(df, unnamed_column_list)

        return df
    
    @staticmethod
    def add_column(df: pd.DataFrame, column_name: str, value: any = np.nan) -> pd.DataFrame:
        """
        Add a column to a DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to add column
        column_name : str
            Name of the column to add
        value : any, optional
            Value to add to the column

        Returns
        -------
        pd.DataFrame
            DataFrame with added column
        """
        logger.info(f'Add column {column_name} to DataFrame')
        if column_name not in df.columns:
            df[column_name] = pd.Series([value] * len(df))
        return df

    @staticmethod
    def set_value(df: pd.DataFrame, column_name: str, indexs: list[int], value: any) -> pd.DataFrame:
        """
        Set a value in a DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to set value
        column_name : str
            Name of the column to set value
        index : list of int
            Indexs to set value
        value : any
            Value to set

        Returns
        -------
        pd.DataFrame
            DataFrame with set value
        """
        logger.info(f'Set value in DataFrame')
        for index in indexs:
            df.at[index, column_name] = value
        return df

    @staticmethod
    def format_dict(data: dict) -> dict:
        """
        Format dictionary keys to lower case and strip

        Parameters
        ----------
        data : dict
            Dictionary to format

        Returns
        -------
        dict
            Formatted dictionary
        """
        logger.info('Format dictionary keys to lower case and strip')
        return {k.strip(): v.strip().upper() for k, v in data.items() if v is not None}
    
    @staticmethod
    def image_to_binary(fig: plt.Figure) -> bytes:
        """
        Converts a matplotlib figure to binary data.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to convert.

        Returns
        -------
        bytes
            The binary representation of the figure.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='PNG')
        buf.seek(0)
        image = Image.open(buf)
        image_byte_array = io.BytesIO()
        image.save(image_byte_array, format='PNG')
        return image_byte_array.getvalue()
    
    @staticmethod
    def binary_to_image(data: bytes) -> plt.Figure:
        """
        Converts binary data to a matplotlib figure.

        Parameters
        ----------
        data : bytes
            The binary data to convert.

        Returns
        -------
        matplotlib.figure.Figure
            The figure representation of the binary data.
        """
        image = Image.open(io.BytesIO(data))
        fig = plt.figure()
        plt.imshow(image)
        return fig
    
    @staticmethod
    def timestamp_to_int(timestamp: pd.Timestamp, tz_info = Const.DEFAULT_TIME_ZONE) -> int:
        """
        Function to convert a timestamp to a Unix timestamp

        Parameters
        ----------
        timestamp : pd.Timestamp
            Timestamp to convert
        tz_info : timezone, optional
            Timezone info, by default Const.DEFAULT_TIME_ZONE

        Returns
        -------
        int
            Unix timestamp
        """
        try:
            dt = timestamp.to_pydatetime()
        except ValueError:
            logger.error(f'Invalid timestamp: {timestamp}')

        return int(dt.timestamp())

    
    @staticmethod
    def time_str_series_to_seconds(t_series: pd.Series) -> np.ndarray:
        """
        Convert a pandas Series of time strings in "HH:MM:SS.fff" format to an array of integers representing seconds.

        Parameters:
        - t_series (pd.Series): A pandas Series containing time strings.

        Returns:
        - np.ndarray: An array of integers representing the time in milliseconds.
        """
        # Convert the time strings to Timedelta
        t_timedelta = pd.to_timedelta(t_series)
        # Convert the Timedelta to total seconds and cast to int64
        t_milliseconds = (t_timedelta.dt.total_seconds()).astype('int64')
        # Convert to numpy array
        t_array = t_milliseconds.to_numpy()
        
        return t_array
    
    @staticmethod
    def datetime_series_to_unix_timestamps(dt_series: pd.Series) -> np.ndarray:
        """
        Convert a pandas Series of datetime objects to an array of Unix timestamps.

        Parameters:
        - dt_series (pd.Series): A pandas Series containing datetime objects.

        Returns:
        - pd.Series: A pandas Series containing Unix timestamps.
        """
        # Convert the datetime objects to Unix timestamps
        unix_timestamps = dt_series.apply(lambda x: Utils.datetime_to_unix_timestamp(x) * 1000)
        return unix_timestamps.to_numpy(dtype=np.float64)
    

    @staticmethod
    def time_string_to_seconds(time_string: str) -> int:
        """
        Converts a time string in the format HH:MM:SS.mmm to seconds.
        
        Parameters:
        - time_string (str): The time string to convert.
        
        Returns:
        - int: The number of seconds represented by the time string.
        """
        hours, minutes, seconds = map(float, time_string.split(":"))
        total_seconds = int(hours * 3600 + minutes * 60 + seconds)
        return total_seconds

    @staticmethod
    def date_str_to_unix_timestamp(date_str: str) -> int:
        """
        Convert a date string like '1/1/2024' to a Unix timestamp in milliseconds.

        Parameters:
        - date_str (str): A string representing a date, e.g. '1/1/2024' or '01/01/2024'.

        Returns:
        - int: Unix timestamp in milliseconds.
        """
        # Parse the date string into a datetime object
        dt = datetime.strptime(date_str, '%m/%d/%Y')
        
        # Convert to Unix timestamp in milliseconds
        unix_ts_ms = int(dt.timestamp() * 1000)
        
        return unix_ts_ms

    @staticmethod
    def datetime_to_unix_timestamp(dt: datetime) -> int:
        """
        Converts a datetime object to a Unix timestamp.

        Parameters:
        - dt (datetime): The datetime object to convert.

        Returns:
        - int: The Unix timestamp.
        """
        return int(dt.timestamp())
    
    @staticmethod
    def search_files(base_path: str, keyword: str, file_extensions: list[str]) -> list[str]:
        """
        Search for files in a directory with a keyword and specific file extensions.

        Parameters:
        - base_path (str): The directory to search in.
        - keyword (str): The keyword to search for in the file names.
        - file_extensions (list[str]): A list of file extensions to search for.

        Returns:
        - list[str]: A list of file paths that match the search criteria.
        """
        logger.info(f'Search for files in {base_path} with keyword: {keyword} and file extensions: {file_extensions}')
        files = []
        for root, dirs, filenames in os.walk(base_path):
            for filename in filenames:
                if not filename.startswith('~$') and keyword.lower() in filename.lower() and filename.split('.')[-1] in file_extensions:
                    files.append(os.path.join(root, filename))
        return files
    
    @staticmethod
    def backup_file(file_path):
        """
        Create a backup of a file by copying it and adding '(copy)' to the name.

        Parameters:
        - file_path (str): The path to the file to backup.

        Returns:
        - str: The path to the backup   
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")
        
        file_dir, file_name = os.path.split(file_path)
        name, ext = os.path.splitext(file_name)
        backup_name = f"{name}(copy){ext}"
        backup_path = os.path.join(file_dir, backup_name)
        
        shutil.copyfile(file_path, backup_path)
        return backup_path
    
    @staticmethod
    def restore_file(backup_path):
        """
        Restore a file from a backup by removing '(copy)' from the name.

        Parameters:
        - backup_path (str): The path to the backup file.

        Returns:
        - str: The path to the restored file
        """
        if not os.path.isfile(backup_path):
            raise FileNotFoundError(f"No such file: '{backup_path}'")
        
        file_dir, file_name = os.path.split(backup_path)
        name, ext = os.path.splitext(file_name)
        if name.endswith("(copy)"):
            original_name = name[:-6]  # Remove (copy)
            original_path = os.path.join(file_dir, f"{original_name}{ext}")
            
            shutil.copyfile(backup_path, original_path)
            os.remove(backup_path)
            return original_path
        else:
            raise ValueError(f"File name does not end with '(copy)': '{file_name}'")

    @staticmethod    
    def delete_backups(directory, file_extension='.res'):
        """
        Traverse the specified directory and delete backup files that end with '(copy)'
        if the corresponding original files exist. Optionally, filter by file extension.

        Parameters:
        - directory (str): The path to the directory to traverse.
        - file_extension (str, optional): The file extension to filter by (e.g., '.txt').

        Returns:
        - int: The number of deleted backup files.
        """
        deleted_count = 0

        for root, _, files in os.walk(directory):
            for file_name in files:
                name, ext = os.path.splitext(file_name)
                if name.endswith("(copy)") and ext == file_extension:
                    backup_path = os.path.join(root, file_name)
                    original_name = name[:-6]  # Remove (copy)
                    original_path = os.path.join(root, f"{original_name}{ext}")

                    if os.path.isfile(original_path):
                        os.remove(backup_path)
                        deleted_count += 1
                        print(f"Deleted backup file: {backup_path}")

        return deleted_count
