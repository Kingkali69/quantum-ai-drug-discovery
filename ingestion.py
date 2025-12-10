"""
Molecular Data Ingestion Pipeline
Professional-grade molecular data processing and feature generation.

Author: KK&GDevOps LLC - Kingkali
Built for enterprise-level drug discovery applications.
"""

import os
import logging
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import gzip
import zipfile
import tempfile
from dataclasses import dataclass

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, SmilesMolSupplier
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import torch
from torch_geometric.data import Data
import deepchem as dc
from tqdm.auto import tqdm

from ..database.manager import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class MolecularDataset:
    """Container for molecular dataset information."""
    name: str
    description: str
    task_type: str  # 'classification' or 'regression'
    url: Optional[str] = None
    file_format: str = 'csv'  # 'csv', 'sdf', 'pdb'
    smiles_column: str = 'smiles'
    label_column: str = 'label'
    
    
class MolecularDataIngestion:
    """
    Production-grade molecular data ingestion pipeline.
    
    Features:
    - Multiple file format support (SMILES, SDF, PDB)
    - Automated feature generation (fingerprints, descriptors, graphs)
    - Data validation and cleaning
    - Progress tracking and error recovery
    - Database integration with batch processing
    """
    
    def __init__(self, db_manager: DatabaseManager, data_dir: str = "./data"):
        """
        Initialize the ingestion pipeline.
        
        Args:
            db_manager: Database manager instance
            data_dir: Directory for storing downloaded data
        """
        self.db_manager = db_manager
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Predefined datasets
        self.datasets = {
            'BBBP': MolecularDataset(
                name='BBBP',
                description='Blood-Brain Barrier Penetration',
                task_type='classification',
                url='https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
                smiles_column='smiles',
                label_column='p_np'
            ),
            'Tox21': MolecularDataset(
                name='Tox21',
                description='Toxicity in 21st Century',
                task_type='classification',
                url='https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
                smiles_column='smiles',
                label_column='NR-AR'
            ),
            'ESOL': MolecularDataset(
                name='ESOL',
                description='Estimated SOLubility',
                task_type='regression',
                url='https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
                smiles_column='smiles',
                label_column='measured log solubility in mols per litre'
            ),
            'FreeSolv': MolecularDataset(
                name='FreeSolv',
                description='Free Solvation Database',
                task_type='regression',
                url='https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
                smiles_column='smiles',
                label_column='expt'
            )
        }
        
        logger.info(f"MolecularDataIngestion initialized with {len(self.datasets)} predefined datasets")
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> Path:
        """Download a predefined dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")
        
        dataset = self.datasets[dataset_name]
        if not dataset.url:
            raise ValueError(f"No URL provided for dataset {dataset_name}")
        
        filename = dataset.url.split('/')[-1]
        file_path = self.data_dir / filename
        
        if file_path.exists() and not force_download:
            logger.info(f"Dataset {dataset_name} already exists at {file_path}")
            return file_path
        
        logger.info(f"Downloading {dataset_name} from {dataset.url}")
        
        try:
            with tqdm(unit='B', unit_scale=True, desc=f"Downloading {filename}") as pbar:
                def progress_hook(block_num, block_size, total_size):
                    pbar.total = total_size
                    pbar.update(block_size)
                
                urllib.request.urlretrieve(dataset.url, file_path, progress_hook)
            
            logger.info(f"Successfully downloaded {dataset_name} to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {str(e)}")
            if file_path.exists():
                file_path.unlink()
            raise
    
    def load_dataset(self, dataset_name: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load and preprocess a dataset."""
        dataset = self.datasets[dataset_name]
        file_path = self.download_dataset(dataset_name)
        
        logger.info(f"Loading dataset {dataset_name} from {file_path}")
        
        try:
            if file_path.suffix == '.gz':
                df = pd.read_csv(file_path, compression='gzip')
            else:
                df = pd.read_csv(file_path)
            
            logger.info(f"Loaded {len(df)} rows from {dataset_name}")
            
            if dataset.smiles_column not in df.columns:
                raise ValueError(f"SMILES column '{dataset.smiles_column}' not found in dataset")
            
            if dataset.label_column not in df.columns:
                available_cols = df.columns.tolist()
                logger.warning(f"Label column '{dataset.label_column}' not found. Available columns: {available_cols}")
                if dataset_name == 'Tox21':
                    task_columns = [col for col in df.columns if col.startswith('NR-') or col.startswith('SR-')]
                    if task_columns:
                        dataset.label_column = task_columns[0]
                        logger.info(f"Using {dataset.label_column} as label column for Tox21")
            
            df = self._preprocess_dataframe(df, dataset)
            
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                logger.info(f"Sampled {sample_size} molecules for testing")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
            raise
    
    def _preprocess_dataframe(self, df: pd.DataFrame, dataset: MolecularDataset) -> pd.DataFrame:
        """Preprocess a molecular dataset DataFrame."""
        logger.info("Preprocessing molecular dataset")
        
        df = df.rename(columns={
            dataset.smiles_column: 'smiles',
            dataset.label_column: 'label'
        })
        
        initial_size = len(df)
        df = df.dropna(subset=['smiles'])
        df = df.drop_duplicates(subset=['smiles'])
        
        valid_mask = df['smiles'].apply(self._is_valid_smiles)
        df = df[valid_mask].reset_index(drop=True)
        
        logger.info(f"Preprocessing complete: {initial_size} -> {len(df)} molecules ({len(df)/initial_size*100:.1f}% retained)")
        
        return df
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if a SMILES string is valid."""
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            return mol is not None
        except:
            return False
    
    def generate_molecular_features(self, smiles_list: List[str], 
                                  include_fingerprints: bool = True,
                                  include_descriptors: bool = True,
                                  include_graphs: bool = True) -> Dict[str, Any]:
        """Generate comprehensive molecular features."""
        logger.info(f"Generating molecular features for {len(smiles_list)} molecules")
        
        features = {
            'smiles': smiles_list,
            'valid_mask': []
        }
        
        if include_fingerprints:
            features.update(self._generate_fingerprints(smiles_list))
        
        if include_descriptors:
            features.update(self._generate_descriptors(smiles_list))
        
        if include_graphs:
            features.update(self._generate_graph_data(smiles_list))
        
        return features
    
    def _generate_fingerprints(self, smiles_list: List[str]) -> Dict[str, List]:
        """Generate various types of molecular fingerprints."""
        logger.info("Generating molecular fingerprints")
        
        morgan_fps = []
        rdkit_fps = []
        maccs_fps = []
        
        for smiles in tqdm(smiles_list, desc="Generating fingerprints"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                morgan_fps.append(np.zeros(2048, dtype=np.uint8))
                rdkit_fps.append(np.zeros(2048, dtype=np.uint8))
                maccs_fps.append(np.zeros(167, dtype=np.uint8))
                continue
            
            morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            morgan_fps.append(np.array(morgan_fp, dtype=np.uint8))
            
            from rdkit.Chem.Fingerprints import FingerprintMols
            rdkit_fp = FingerprintMols.FingerprintMol(mol)
            rdkit_bits = np.zeros(2048, dtype=np.uint8)
            fp_bits = np.array(rdkit_fp, dtype=np.uint8)
            rdkit_bits[:min(len(fp_bits), 2048)] = fp_bits[:min(len(fp_bits), 2048)]
            rdkit_fps.append(rdkit_bits)
            
            from rdkit.Chem import MACCSkeys
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            maccs_fps.append(np.array(maccs_fp, dtype=np.uint8))
        
        return {
            'morgan_fingerprints': np.array(morgan_fps),
            'rdkit_fingerprints': np.array(rdkit_fps),
            'maccs_fingerprints': np.array(maccs_fps)
        }
    
    def ingest_dataset(self, dataset_name: str, sample_size: Optional[int] = None,
                      generate_features: bool = True) -> Dict[str, Any]:
        """Complete ingestion pipeline for a dataset."""
        logger.info(f"Starting complete ingestion pipeline for {dataset_name}")
        
        df = self.load_dataset(dataset_name, sample_size)
        
        logger.info("Storing molecules in database")
        molecules_data = []
        for _, row in df.iterrows():
            mol_data = {'smiles': row['smiles']}
            if 'label' in row:
                mol_data['label'] = row['label']
            molecules_data.append(mol_data)
        
        molecule_ids = self.db_manager.batch_insert_molecules(molecules_data)
        
        if generate_features:
            logger.info("Generating molecular fingerprints")
            fingerprints_count = self.db_manager.generate_molecular_fingerprints(molecule_ids)
        
        if 'label' in df.columns:
            logger.info("Storing bioactivity data")
            bioactivity_data = []
            for _, row in df.iterrows():
                bio_data = {
                    'smiles': row['smiles'],
                    'label_value': row['label'] if pd.notna(row['label']) else None,
                    'label_binary': int(row['label']) if pd.notna(row['label']) and self.datasets[dataset_name].task_type == 'classification' else None,
                    'activity_type': 'bioactivity'
                }
                bioactivity_data.append(bio_data)
            
            bioactivity_count = self.db_manager.store_bioactivity_data(dataset_name, bioactivity_data)
        
        results = {
            'dataset_name': dataset_name,
            'molecules_ingested': len(molecule_ids),
            'fingerprints_generated': fingerprints_count if generate_features else 0,
            'bioactivities_stored': bioactivity_count if 'label' in df.columns else 0,
            'molecule_ids': molecule_ids
        }
        
        logger.info(f"Ingestion complete for {dataset_name}: {results}")
        return results
