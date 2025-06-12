import pandas as pd
import numpy as np
from scipy import integrate
from datetime import datetime

from batteryabn import logger, Constants as Const
from batteryabn.utils import Utils

def create_formatter(timezone: str = None):
    return Formatter(timezone)

class Formatter:
    def __init__(self, timezone: str = None):
        """
        A class to format battery test data.

        Parameters
        ----------
        timezone : str, optional
            Timezone to use for formatting
        """
        # Default NY timezone
        self.timezone = timezone if timezone else Const.DEFAULT_TIME_ZONE

        self.test_data = pd.DataFrame(dtype=object)
        self.metadata = {}
        self.cell_name = None # Cell name for the test data
        self.start_time = None # Start timestamp for the test data
        self.last_update_time = None # Last update timestamp for the test data
        self.calibration_parameters = {}

    def format_data(self, data: pd.DataFrame, metadata: pd.DataFrame, test_type: str) -> pd.DataFrame:
        """
        Format battery test data and metadata.

        Parameters
        ----------
        data : pd.DataFrame
            Raw battery test data

        metadata : dict
            Raw battery test metadata

        test_type : str
            Type of battery test data. Supported types: 'Arbin', 'BioLogic', 'Neware', 'Vdf'

        Returns
        -------
        pd.DataFrame
            Formatted battery test data
        """
        self.clear()
        self.format_metadata(metadata)
        self.format_test_data(data, test_type)

        return self.test_data

    def format_test_data(self, data: pd.DataFrame, test_type: str) -> None:
        """
        Format battery test data.

        Parameters
        ----------
        data : pd.DataFrame
            Raw battery test data

        test_type : str
            Type of battery test data. Supported types: 'Arbin', 'BioLogic', 'Neware', 'Vdf'

        is_cycle : bool, optional
            Whether the data is cycle data
        """
        logger.info('Format battery test data')

        if data is None or data.empty:
            return
        
        df = data.copy()

        df = Utils.drop_unnamed_columns(df)
        df = Utils.drop_empty_rows(df)
        df = Utils.formate_columns(df)

        # Rename columns based on test type
        rename_dict = getattr(Const, f'{test_type.upper()}_RENAME_DICT')
        df = Utils.rename_columns(df, rename_dict)

        # Format the data based on test type
        if test_type != Const.VDF:
            Utils.add_column(df, Const.TEMPERATURE)
            Utils.add_column(df, Const.STEP_IDX)
            Utils.add_column(df, Const.AHT)
        else:
            # Get the Calibration Parameters X1, X2 and C from the cell and start & removal time
            # TODO: Better way to get calibration parameters instead of using sanity check csv file 
            df = self.add_calibration_parameters(df)

        # Calculate AHT from integrating current
        time_reset = df[Const.TIMESTAMP].reset_index(drop=True)
        if not isinstance(time_reset[0], pd.Timestamp):
            time_reset = pd.to_datetime(time_reset, unit='ms')
        current_reset = df[Const.CURRENT].reset_index(drop=True)
        aht_calculated = integrate.cumtrapz(
            abs(current_reset), 
            (time_reset - time_reset[0]).dt.total_seconds(),
        ) / 3600
        aht_calculated = np.append(aht_calculated, aht_calculated[-1] if len(aht_calculated) > 0 else 0)
        df[Const.AHT] = aht_calculated
        logger.info(f'AHT calculated: {df[Const.AHT].iloc[-1]}')

        if test_type == Const.BIOLOGIC:
            if(max(abs(df[Const.CURRENT]))>20): 
                # current data is ma vs A divide by 1000.
                df[Const.CURRENT] = df[Const.CURRENT]/1000
                df[Const.AHT] = df[Const.AHT]/1000 

        elif test_type == Const.NEWARE:
            if df[Const.TEMPERATURE] is not None:
                df[Const.TEMPERATURE] = np.where((df[Const.TEMPERATURE] >= 200) & (df[Const.TEMPERATURE] <250), np.nan, df[Const.TEMPERATURE]) 
            
            # Formate the time column from HH:MM:SS.fff to seconds
            df[Const.TIME] = Utils.time_str_series_to_seconds(df[Const.TIME])

        # Check the data lengths for cycle data
        if test_type != Const.VDF:
            lengths = [len(df[column].reset_index(drop=True)) for column in [Const.TIME, Const.CURRENT, Const.VOLTAGE]]
            if len(set(lengths)) > 1:
                raise ValueError(f"Inconsistent data lengths in the dataframe")

        if not df.empty and Const.TIMESTAMP in df.columns:
            start_timestamp_value, last_timestamp_value = df[Const.TIMESTAMP].iloc[0], df[Const.TIMESTAMP].iloc[-1]
            if isinstance(last_timestamp_value, pd.Timestamp):
                self.start_time = Utils.timestamp_to_int(start_timestamp_value)
                self.last_update_time = Utils.timestamp_to_int(last_timestamp_value)
            else:
                self.start_time = int(start_timestamp_value)
                self.last_update_time = int(last_timestamp_value)

        self.test_data = df

    def format_metadata(self, metadata: dict) -> None:
        """
        Format battery test metadata.

        Parameters
        ----------
        metadata : dict
            Raw battery test metadata
        """
        logger.info('Format battery test metadata')

        metadata = Utils.format_dict(metadata)
        self.metadata = metadata

        # Get cell name from metadata
        cell_name = metadata.get('Project Name') + '_' + metadata.get('Cell ID')
        self.cell_name = cell_name

    def format_calibration_parameters(self, calibration_parameters: dict) -> None:
        """
        Format calibration parameters.

        Parameters
        ----------
        calibration_parameters : dict
            Calibration parameters
        """
        self.calibration_parameters = calibration_parameters

        # Not use time dict to store the calibration parameters at the moment
        # formatted_parameters = {}

        # for cell_name, periods in calibration_parameters.items():
        #     # Sort the periods by start date
        #     periods.sort(key=lambda x: datetime.strptime(x[0], "%m/%d/%Y"))

        #     time_dict = {}

        #     for start, end, x1, x2, c in periods:
        #         #TODO: Use timestamp or datetime object instead of string
        #         start_time = datetime.strptime(start, "%m/%d/%Y")
        #         start_time = Utils.datetime_to_unix_timestamp(start_time)
        #         end_time = datetime.strptime(end, "%m/%d/%Y")
        #         end_time = Utils.datetime_to_unix_timestamp(end_time)

        #         # Get the existing overlapping keys
        #         overlapping_keys = [k for k in time_dict.keys() if k <= start_time]

        #         # Remove the overlapping keys
        #         for key in overlapping_keys:
        #             if key < end_time:
        #                 del time_dict[key]
        #         # Add the new key
        #         time_dict[start_time] = (x1, x2, c)
        #         time_dict[end_time] = (Const.X1, Const.X2, Const.C)

        #     formatted_parameters[cell_name] = time_dict

        # self.calibration_parameters = formatted_parameters

    def add_calibration_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calibration parameters to the test data.

        Parameters
        ----------
        df : pd.DataFrame
            Test data

        Returns
        -------
        pd.DataFrame
            Test data with calibration parameters
        """
        cell_name = self.cell_name
        calibration_parameters = self.calibration_parameters.get(cell_name, {})

        Utils.add_column(df, Const.CALIBRATION_X1, Const.X1)
        Utils.add_column(df, Const.CALIBRATION_X2, Const.X2)
        Utils.add_column(df, Const.CALIBRATION_C, Const.C)

        if not calibration_parameters:
            return df

        #TODO: Check this hard coded condition
        # Arbin already calibrated, LDC N==100 in vdf parameter
        ldc_n = int(df['ldc n'].iloc[0])
        if ldc_n != 100:
            for calibration_parameter in calibration_parameters:
                for protocol, parameters in calibration_parameter.items():
                    start_date, removal_date, x1, x2, c = parameters
                    if start_date is not None and (
                        (self.metadata.get('Test Type') == '_F' and protocol == 'Formation') or
                        (self.metadata.get('Test Type') != '_F' and protocol != 'Formation')
                    ):
                        start_timestamp, removal_timestamp = Utils.date_str_to_unix_timestamp(start_date), Utils.date_str_to_unix_timestamp(removal_date)
                        # Apply calibration only within the specified date range
                        mask = (df[Const.TIMESTAMP] >= start_timestamp) & (df[Const.TIMESTAMP] <= removal_timestamp)
                        logger.info(f'Mask for calibration data: {mask}')
                        df.loc[mask, Const.CALIBRATION_X1] = x1
                        df.loc[mask, Const.CALIBRATION_X2] = x2
                        df.loc[mask, Const.CALIBRATION_C] = c

        return df

    def clear(self) -> None:
        """
        Clear formatted data and metadata
        """
        self.test_data = pd.DataFrame(dtype=object)
        self.metadata = {}
        self.cell_name = None