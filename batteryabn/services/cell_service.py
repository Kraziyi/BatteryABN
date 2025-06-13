from flask_injector import inject
from batteryabn import logger, Constants as Const
from batteryabn.models import Cell, Project
from batteryabn.repositories import CellRepository, TestRecordRepository, ProjectRepository, FileSystemRepository
from batteryabn.repositories import create_cell_repository, create_test_record_repository, create_project_repository, create_filesystem_repository
from batteryabn.utils import Processor, Viewer, Utils

def create_cell_service(session=None):
    cell_repository = create_cell_repository(session)
    test_record_repository = create_test_record_repository(session)
    project_repository = create_project_repository(session)
    filesystem_repository = create_filesystem_repository()
    return CellService(cell_repository, test_record_repository, project_repository, filesystem_repository)

class CellService:
    """
    The CellService class provides an interface for creating and querying Cell objects.
    """
    @inject
    def __init__(self, cell_repository: CellRepository, test_record_repository: TestRecordRepository, 
                 project_repository: ProjectRepository, filesystem_repository: FileSystemRepository):
        self.cell_repository = cell_repository
        self.test_record_repository = test_record_repository
        self.project_repository = project_repository
        self.filesystem_repository = filesystem_repository


    def create_cell(self, cell_name: str):
        """
        This method creates a new Cell instance and adds it to the session.

        Parameters
        ----------
        cell_name : str
            The unique name of the cell.

        Returns
        -------
        Cell
            The newly created Cell object
        """
        cell = self.cell_repository.find_by_name(cell_name)
        if cell:
            logger.info(f'Found existing cell: {cell_name}')
            return cell

        cell = Cell(cell_name=cell_name.upper())  # Assuming cell names are case-insensitive and stored as uppercase
        self.cell_repository.add(cell)
        try:
            self.cell_repository.commit()
            logger.info(f'Created new cell: {cell_name}')
        except Exception as e:
            self.cell_repository.rollback()
            logger.error(f'Failed to create new cell: {cell_name}. Error: {e}')
            raise

        return cell
    
    def process_cell(self, cell_name: str, processor: Processor, viewer: Viewer, test_type: str = None):
        """
        Process cell data and save it to the database.

        Parameters
        ----------
        cell : Cell
            The cell to process
        processor : Processor
            Processor object with processed cell data
        viewer : Viewer
            Viewer object with plotting functions
        test_type : str, optional
            The type of test to process, by default None
        """
        cell = self.find_cell_by_name(cell_name)
        if not cell:
            logger.error(f'Cell not found: {cell_name}')
            return
        
        project = cell.project
        cycler_trs, vdf_trs = self.get_cycler_vdf_trs(cell, test_type)        
        # Process cell data
        processor.process(cycler_trs, vdf_trs, cell.project)
        if processor.cell_data.empty:
            logger.error(f'No data found for cell: {cell_name}')
            return
        # Genrate images for processed data
        img_cell, img_ccm, img_ccm_aht, img_cell_html, img_ccm_html, img_ccm_aht_html = viewer.plot(processor.cell_data, processor.cell_cycle_metrics, processor.cell_data_vdf, cell_name)

        cell.calibration_c = processor.cell_data_vdf.iloc[-1][Const.CALIBRATION_C]
        cell.calibration_x1 = processor.cell_data_vdf.iloc[-1][Const.CALIBRATION_X1]
        cell.calibration_x2 = processor.cell_data_vdf.iloc[-1][Const.CALIBRATION_X2]
        # # Update cell data
        # cell.cell_data = Utils.gzip_pikle_dump(processor.cell_data)
        # cell.cell_cycle_metrics = Utils.gzip_pikle_dump(processor.cell_cycle_metrics)
        # cell.cell_data_vdf = Utils.gzip_pikle_dump(processor.cell_data_vdf)

        # cell.image_cell = Utils.image_to_binary(img_cell)
        # cell.image_ccm = Utils.image_to_binary(img_ccm)
        # cell.image_ccm_aht = Utils.image_to_binary(img_ccm_aht)

        # Save cell data
        # try:
        #     self.cell_repository.commit()
        #     logger.info(f'Processed and saved data for cell: {cell_name}')
        # except Exception as e:
        #     self.cell_repository.rollback()
        #     logger.error(f'Failed to save processed data for cell: {cell_name}. Error: {e}')
        #     raise e
        
        # Save data to local file
        self.filesystem_repository.save_df_to_csv(project.project_name, cell.cell_name, 'cell_cycle_metrics', processor.cell_cycle_metrics)
        self.filesystem_repository.save_to_local_pklgz(project.project_name, cell.cell_name, 'cell_data', processor.cell_data)
        self.filesystem_repository.save_to_local_pklgz(project.project_name, cell.cell_name, 'cell_cycle_metrics', processor.cell_cycle_metrics)
        self.filesystem_repository.save_to_local_pklgz(project.project_name, cell.cell_name, 'cell_data_vdf', processor.cell_data_vdf)
        self.filesystem_repository.save_to_local_pklgz(project.project_name, cell.cell_name, 'cell_data_rpt', processor.cell_data_rpt)
        self.filesystem_repository.save_plt_to_png(project.project_name, cell.cell_name, 'cell', img_cell)
        self.filesystem_repository.save_plt_to_png(project.project_name, cell.cell_name, 'ccm', img_ccm)
        self.filesystem_repository.save_plt_to_png(project.project_name, cell.cell_name, 'ccm_aht', img_ccm_aht)    
        self.filesystem_repository.save_html(project.project_name, cell.cell_name, 'cell', img_cell_html)
        self.filesystem_repository.save_html(project.project_name, cell.cell_name, 'ccm', img_ccm_html)
        self.filesystem_repository.save_html(project.project_name, cell.cell_name, 'ccm_aht', img_ccm_aht_html)

    def process_cells_for_project(self, project_name: str, processor: Processor, viewer: Viewer):
        """
        Process cells for a project.

        Parameters
        ----------
        project_name : str
            The name of the project to process cells for
        processor : Processor
            Processor object with processed cell data
        """
        cells = self.find_cells_by_project_name(project_name)
        if not cells:
            logger.error(f'No cells found for project: {project_name}')
            return
        
        for cell in cells:
            try:
                self.process_cell(cell.cell_name, processor, viewer)
            except Exception as e:
                logger.error(f'Failed to process cell: {cell.cell_name}. Error: {e}')
                continue

    def generate_cell_images_by_processed_data(self, cell_name: str, viewer: Viewer):
        """
        Generate images for a cell based on processed data.
        Find the processed data for the cell and generate images based on it.

        Parameters
        ----------
        cell_name : str
            The unique name of the cell.
        viewer : Viewer
            Viewer object with plotting functions

        Returns
        -------
        tuple
            The images for the cell
        """
        cell_data, cell_cycle_metrics, cell_data_vdf, _ = self.get_processed_data(cell_name)
        if not cell_data or not cell_cycle_metrics or not cell_data_vdf:
            logger.error(f'Processed data not found for cell: {cell_name}')
            return
        
        img_cell, img_ccm, img_ccm_aht, img_cell_html, img_ccm_html, img_ccm_aht_html = viewer.plot(cell_data, cell_cycle_metrics, cell_data_vdf, cell_name)
        return img_cell, img_ccm, img_ccm_aht
            
        
    def get_processed_data(self, cell_name: str):
        """
        Get processed data for a cell.

        Parameters
        ----------
        cell_name : str
            The unique name of the cell.

        Returns
        -------
        tuple
            The processed data for the cell
        """
        cell = self.find_cell_by_name(cell_name)
        if not cell:
            logger.error(f'Cell not found: {cell_name}')
            return None, None, None
        
        # TODO: This should be used in the future when the data is stored in the database
        # return cell.load_cell_data(), cell.load_cell_cycle_metrics(), cell.load_cell_data_vdf()

        project = cell.project
        cell_data = self.filesystem_repository.load_from_local_pklgz(project.project_name, cell.cell_name, 'cell_data')
        cell_cycle_metrics = self.filesystem_repository.load_from_local_pklgz(project.project_name, cell.cell_name, 'cell_cycle_metrics')
        cell_data_vdf = self.filesystem_repository.load_from_local_pklgz(project.project_name, cell.cell_name, 'cell_data_vdf')
        cell_data_rpt = self.filesystem_repository.load_from_local_pklgz(project.project_name, cell.cell_name, 'cell_data_rpt')

        return cell_data, cell_cycle_metrics, cell_data_vdf, cell_data_rpt

    def get_data(self, cell_name: str, data_type: str):
        """
        Get cell data for a cell.

        Parameters
        ----------
        cell_name : str
            The unique name of the cell.
        data_type : str
            The type of data to get

        Returns
        -------
        dict
            The cell data
        """
        cell = self.find_cell_by_name(cell_name)
        if not cell:
            logger.error(f'Cell not found: {cell_name}')
            return None
        
        # TODO: This should be used in the future when the data is stored in the database

        project = cell.project
        try:
            cell_data = self.filesystem_repository.load_from_local_pklgz(project.project_name, cell.cell_name, data_type)
        except Exception as e:
            logger.error(f'Failed to load data for cell: {cell_name}. Error: {e}')
            return None
        return cell_data

    def load_cell_images(self, cell_name: str):
        """
        Load cell images from the database.

        Parameters
        ----------
        cell_name : str
            The unique name of the cell.

        Returns
        -------
        tuple
            The images for the cell
        """
        cell = self.find_cell_by_name(cell_name)

        if not cell:
            logger.error(f'Cell not found: {cell_name}')
            return
        
        return cell.load_image_cell(), cell.load_image_ccm(), cell.load_image_ccm_aht()

    def get_combined_cell_data(self, cell_name: str, processor: Processor):
        """
        Get combined cell data and cell data vdf for a cell.

        Parameters
        ----------
        cell_name : str
            The unique name of the cell.
        processor : Processor
            Processor object with processed cell data

        Returns
        -------
        dict
            Combined cell data
        """
        cell_data, _, cell_data_vdf, _ = self.get_processed_data(cell_name)
        if cell_data is None or cell_data_vdf is None:
            logger.error(f'Processed data not found for cell: {cell_name}')
            return
        
        combined_cell_data = processor.combine_data(cell_data, cell_data_vdf)

        # Save combined data to local file
        project = self.find_cell_by_name(cell_name).project
        self.filesystem_repository.save_to_local_pklgz(project.project_name, cell_name, 'combined_cell_data', combined_cell_data)
        return combined_cell_data


    def change_cell_project(self, cell_name: str, new_project_name: str):
        """
        Change the project associated with a cell to a new project.

        Parameters
        ----------
        cell_name : str
            The unique name of the cell.
        new_project_name : str
            The name of the new project to associate with the cell.
        """
        # Find the cell by name
        cell = self.cell_repository.find_by_name(cell_name)
        if not cell:
            logger.error(f'Cell with name {cell_name} not found.')
            return

        # Find or create the new project
        project = self.project_repository.find_by_name(new_project_name)
        if not project:
            logger.info(f'Creating new project: {new_project_name}')
            project = Project(project_name=new_project_name)
            self.project_repository.add(project)

        # Associate the cell with the new project
        cell.project = project
        logger.info(f'Cell {cell_name} is now associated with project {new_project_name}.')

        # Commit the changes
        try:
            self.cell_repository.commit()
            logger.info(f'Successfully updated project association for cell: {cell_name}.')
        except Exception as e:
            self.cell_repository.rollback()
            logger.error(f'Failed to update project association for cell: {cell_name}. Error: {e}')
            raise


    def get_cycler_vdf_trs(self, cell: Cell, test_type: str = None):
        """
        Get cycler and Vdf test records for a cell.

        Parameters
        ----------
        cell : Cell
            The cell to get test records for
        test_type : str, optional
            The type of test record to get, by default None

        Returns
        -------
        dict
            Cycler test records
        dict
            Vdf test records
        """
        trs = self.test_record_repository.find_by_cell_name(cell.cell_name)
        cycler_trs, vdf_trs = {}, {}
        for tr in trs:
            if tr.last_update_time is None:
                #TODO: Maybe check more details if the test record contains data
                continue
            if tr.test_type == Const.VDF:
                vdf_trs[tr.test_name] = tr
            elif test_type is None or tr.test_type == test_type:
                    cycler_trs[tr.test_name] = tr
            else: 
                logger.info(f'Skipping test record: {tr.test_name}')
        return cycler_trs, vdf_trs

    def find_cell_by_name(self, cell_name: str):
        """
        This method finds a Cell by its name.

        Parameters
        ----------
        cell_name : str
            The name of the cell to find

        Returns
        -------
        Cell
            The Cell object with the specified name
        """
        return self.cell_repository.find_by_name(cell_name)
    
    def find_cells_by_project_name(self, project_name: str):
        """
        This method finds all Cells in a Project.

        Parameters
        ----------
        project_name : str
            The name of the project to find cells for

        Returns
        -------
        list
            A list of Cell objects in the specified project
        """
        return self.cell_repository.find_by_project(project_name)
    
    def find_cells_by_keyword(self, keyword: str):
        """
        This method finds all Cells with a keyword in their name.

        Parameters
        ----------
        keyword : str
            The keyword to search for in cell names

        Returns
        -------
        list
            A list of Cell objects with the specified keyword in their name
        """
        return self.cell_repository.find_by_keyword(keyword)
    
    def delete_cell(self, cell_name: str):
        """
        This method deletes a Cell and all associated TestRecords from the database.

        Parameters
        ----------
        cell_name : str
            The name of the cell to delete
        """

        cell = self.cell_repository.find_by_name(cell_name)
        if not cell:
            logger.info(f'Cell not found: {cell_name}')
            return

        # Delete associated test records
        trs = self.test_record_repository.find_by_cell_name(cell_name)
        for tr in trs:
            self.test_record_repository.delete(tr)
        self.test_record_repository.commit()
        logger.info(f'Deleted test records for cell: {cell_name}')
        
        self.cell_repository.delete(cell)
        self.cell_repository.commit()
        logger.info(f'Deleted cell: {cell_name}')

    def get_cell_imgs_paths(self, cell_name:str):
        """
        Get paths to cell images for a cell.

        Parameters
        ----------
        cell_name : str
            The name of the cell

        Returns
        -------
        tuple
            Paths to cell images
        """

        #TODO: Temporary solution for images saved in local files
        cell = self.find_cell_by_name(cell_name)
        if not cell:
            logger.error(f'Cell not found: {cell_name}')
            return
        project = cell.project
        return self.filesystem_repository.get_cell_imgs_paths(project.project_name, cell_name)
    
    def get_cell_htmls_paths(self, cell_name:str):
        """
        Get paths to cell images for a cell.

        Parameters
        ----------
        cell_name : str
            The name of the cell

        Returns
        -------
        tuple
            Paths to cell images
        """
        cell = self.find_cell_by_name(cell_name)
        if not cell:
            logger.error(f'Cell not found: {cell_name}')
            return
        project = cell.project
        return self.filesystem_repository.get_cell_htmls_paths(project.project_name, cell_name)

    def get_latest_test_record(self, cell_name: str):
        """
        Get the latest test record for a cell.

        Parameters
        ----------
        cell_name : str
            The name of the cell
        test_type : str
            The type of test record to get

        Returns
        -------
        TestRecord
            The latest test record
        """
        trs = self.test_record_repository.find_by_cell_name(cell_name)
        latest_tr = None
        for tr in trs:
            if tr.test_type != Const.VDF and (latest_tr is None or tr.last_update_time > latest_tr.last_update_time):
                latest_tr = tr
        return latest_tr
    
    def get_latest_cell_info(self, cell_name: str):
        """
        Get the latest report info for a cell.

        Parameters
        ----------
        cell_name : str
            The name of the cell

        Returns
        -------
        dict
            The latest report info
        """

        cell_data_rpt = self.get_data(cell_name, 'cell_data_rpt')
        if cell_data_rpt is None:
            return None
        # Get the last row of the data
        rpt_last_row = cell_data_rpt.iloc[-1]
        tr_name = rpt_last_row[Const.TEST_NAME]
        data_last_row = rpt_last_row[Const.DATA].iloc[-1]

        ccm = self.get_data(cell_name, 'cell_cycle_metrics')
        if ccm is None:
            return {"latest_test_name": tr_name, "timestamp": data_last_row[Const.TIMESTAMP], "capacity": data_last_row[Const.AHT]}
        
        # Get the last row of the cell cycle metrics
        ccm_last_row = ccm.iloc[-1]

        return {"latest_test_name": tr_name, "timestamp": data_last_row[Const.TIMESTAMP], "capacity": data_last_row[Const.AHT],
                "protocol": ccm_last_row[Const.PROTOCOL], "cycle_type": ccm_last_row[Const.CYCLE_TYPE]}
    
    #  Load the tr names based on cell name