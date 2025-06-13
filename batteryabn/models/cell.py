from batteryabn.utils import Utils
from ..extensions import db

class Cell(db.Model):
    """
    Class for storing battery cell data.
    """

    __tablename__ = 'cells'

    id = db.Column(db.Integer, primary_key=True)
    cell_name = db.Column(db.String, unique=True)
    # Unique cell name. i.e. 'GMJuly2022_CELL002'
    test_records = db.relationship("TestRecord", back_populates="cell")
    project_name = db.Column(db.String, db.ForeignKey('projects.project_name'))
    project = db.relationship("Project", back_populates="cells")

    #Latest sanity check data for calibration data
    calibration_x1 = db.Column(db.Float, nullable=True)
    calibration_x2 = db.Column(db.Float, nullable=True)
    calibration_c = db.Column(db.Float, nullable=True)

    # Binary data fields
    cell_data = db.Column(db.LargeBinary, nullable=True)
    cell_cycle_metrics = db.Column(db.LargeBinary, nullable=True)
    cell_data_vdf = db.Column(db.LargeBinary, nullable=True)

    # Images for the processed data
    image_cell = db.Column(db.LargeBinary, nullable=True)
    image_ccm = db.Column(db.LargeBinary, nullable=True)
    image_ccm_aht = db.Column(db.LargeBinary, nullable=True)


    def load_cell_data(self):
        """
        Load cell data from the database.
        """
        return Utils.gzip_pickle_load(self.cell_data)
    
    def load_cell_cycle_metrics(self):
        """
        Load cell cycle metrics from the database.
        """
        return Utils.gzip_pickle_load(self.cell_cycle_metrics)
    
    def load_cell_data_vdf(self):
        """
        Load cell VDF data from the database.
        """
        return Utils.gzip_pickle_load(self.cell_data_vdf)  
    
    def load_image_cell(self):
        """
        Load the cell image from the database.
        """
        return Utils.binary_to_image(self.image_cell)

    def load_image_ccm(self):
        """
        Load the cell cycle metrics image from the database.
        """
        return Utils.binary_to_image(self.image_ccm)
    
    def load_image_ccm_aht(self):
        """
        Load the cell cycle metrics AHT image from the database.
        """
        return Utils.binary_to_image(self.image_ccm_aht)
    
    def to_dict(self):
        """
        Convert the Cell object to a dictionary.
        """
        return {
            'cell_name': self.cell_name,
            'project_name': self.project_name,
            # 'cell_data': self.load_cell_data(),
            # 'cell_cycle_metrics': self.load_cell_cycle_metrics(),
            # 'cell_data_vdf': self.load_cell_data_vdf(),
            # 'image_cell': self.load_image_cell(),
            # 'image_ccm': self.load_image_ccm(),
            # 'image_ccm_aht': self.load_image_ccm_aht()
        }