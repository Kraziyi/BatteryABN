"""Initial migration

Revision ID: a2e998a1dbee
Revises: 
Create Date: 2024-03-27 10:17:48.445005

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a2e998a1dbee'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('projects',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('project_name', sa.String(), nullable=True),
    sa.Column('qmax', sa.Float(), nullable=True),
    sa.Column('i_c20', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('project_name')
    )
    op.create_table('cells',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('cell_name', sa.String(), nullable=True),
    sa.Column('project_name', sa.String(), nullable=True),
    sa.Column('cell_data', sa.LargeBinary(), nullable=True),
    sa.Column('cell_cycle_metrics', sa.LargeBinary(), nullable=True),
    sa.Column('cell_data_vdf', sa.LargeBinary(), nullable=True),
    sa.Column('image_cell', sa.LargeBinary(), nullable=True),
    sa.Column('image_ccm', sa.LargeBinary(), nullable=True),
    sa.Column('image_ccm_aht', sa.LargeBinary(), nullable=True),
    sa.ForeignKeyConstraint(['project_name'], ['projects.project_name'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('cell_name')
    )
    op.create_table('testrecords',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('test_name', sa.String(), nullable=True),
    sa.Column('test_type', sa.String(), nullable=True),
    sa.Column('cell_name', sa.String(), nullable=True),
    sa.Column('test_data', sa.LargeBinary(), nullable=True),
    sa.Column('test_metadata', sa.LargeBinary(), nullable=True),
    sa.Column('last_update_time', sa.BIGINT(), nullable=True),
    sa.ForeignKeyConstraint(['cell_name'], ['cells.cell_name'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('test_name')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('testrecords')
    op.drop_table('cells')
    op.drop_table('projects')
    # ### end Alembic commands ###