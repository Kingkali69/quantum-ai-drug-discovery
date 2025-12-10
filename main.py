#!/usr/bin/env python3
"""
Quantum-AI Drug Discovery Framework - Main Entry Point
Professional software architecture demonstrating enterprise-level quantum computing integration.

Author: KK&GDevOps LLC - Kingkali
Built in 40 minutes for a $300K position demonstration
Usage: python main.py [command] [options]
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.database.manager import DatabaseManager
from src.data.ingestion import MolecularDataIngestion
from src.training.pipeline import QuantumTrainingPipeline, TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_database(database_url: str) -> DatabaseManager:
    """Initialize and test database connection."""
    logger.info("🔌 Setting up database connection")
    
    db_manager = DatabaseManager(database_url)
    
    if db_manager.test_connection():
        logger.info("✅ Database connection successful")
        return db_manager
    else:
        logger.error("❌ Database connection failed")
        logger.error("💡 Make sure PostgreSQL is running: docker-compose up -d")
        sys.exit(1)


def ingest_data(db_manager: DatabaseManager, dataset_name: str, sample_size: int = None):
    """Ingest molecular dataset."""
    logger.info(f"📥 Starting data ingestion for {dataset_name}")
    
    ingestion = MolecularDataIngestion(db_manager)
    
    try:
        results = ingestion.ingest_dataset(dataset_name, sample_size=sample_size)
        
        logger.info("✅ Data ingestion completed successfully")
        logger.info(f"   Molecules ingested: {results['molecules_ingested']}")
        logger.info(f"   Fingerprints generated: {results['fingerprints_generated']}")
        logger.info(f"   Bioactivities stored: {results['bioactivities_stored']}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Data ingestion failed: {str(e)}")
        sys.exit(1)


def train_model(db_manager: DatabaseManager, config_dict: Dict[str, Any]):
    """Train a single model."""
    logger.info(f"🚀 Starting model training")
    
    config = TrainingConfig(**config_dict)
    trainer = QuantumTrainingPipeline(config, db_manager)
    
    try:
        results = trainer.train()
        
        logger.info("✅ Model training completed successfully")
        logger.info(f"   Training run ID: {results['training_run_id']}")
        logger.info(f"   Best validation score: {results['best_val_score']:.4f}")
        logger.info(f"   Total epochs: {results['total_epochs']}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}")
        sys.exit(1)


def compare_models(db_manager: DatabaseManager, config_dict: Dict[str, Any], 
                  model_types: list = None):
    """Compare multiple model types."""
    logger.info("🔬 Starting model comparison")
    
    if model_types is None:
        model_types = ['fingerprint_classifier', 'classical_gnn', 'quantum_hybrid']
    
    config = TrainingConfig(**config_dict)
    trainer = QuantumTrainingPipeline(config, db_manager)
    
    try:
        results = trainer.compare_models(model_types)
        
        logger.info("✅ Model comparison completed successfully")
        logger.info("\n🏆 RESULTS SUMMARY:")
        logger.info("=" * 50)
        
        for model_type, model_results in results.items():
            if 'error' not in model_results:
                test_metrics = model_results['final_metrics']['test']
                logger.info(f"{model_type}: AUC = {test_metrics['auc']:.4f}, "
                           f"Accuracy = {test_metrics['accuracy']:.4f}")
            else:
                logger.info(f"{model_type}: FAILED - {model_results['error']}")
        
        # Determine winner
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model = max(valid_results.keys(), 
                           key=lambda x: valid_results[x]['final_metrics']['test']['auc'])
            best_auc = valid_results[best_model]['final_metrics']['test']['auc']
            
            logger.info(f"\n🎉 WINNER: {best_model} (AUC = {best_auc:.4f})")
            
            if best_model == 'quantum_hybrid':
                logger.info("🔮 QUANTUM SUPREMACY ACHIEVED!")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Model comparison failed: {str(e)}")
        sys.exit(1)


def run_full_pipeline(database_url: str, dataset_name: str = 'BBBP', 
                     sample_size: int = 1000, quick_mode: bool = False):
    """Run the complete pipeline from ingestion to model comparison."""
    logger.info("🚀 Starting FULL QUANTUM-AI PIPELINE")
    logger.info("=" * 60)
    logger.info("KK&GDevOps Quantum-AI Demonstration")
    logger.info("Built in 40 minutes - Enterprise-Level Architecture")
    logger.info("=" * 60)
    
    # Setup database
    db_manager = setup_database(database_url)
    
    # Ingest data
    logger.info(f"\n📥 PHASE 1: Data Ingestion ({dataset_name})")
    logger.info("-" * 40)
    ingest_data(db_manager, dataset_name, sample_size)
    
    # Training configuration
    config_dict = {
        'dataset_name': dataset_name,
        'hidden_dim': 64 if quick_mode else 128,
        'num_layers': 2 if quick_mode else 3,
        'batch_size': 16 if quick_mode else 32,
        'num_epochs': 5 if quick_mode else 20,
        'early_stopping_patience': 3 if quick_mode else 10,
        'quantum_dim': 16 if quick_mode else 32,
        'num_qubits': 4,
        'quantum_depth': 1 if quick_mode else 2,
    }
    
    # Model comparison
    logger.info(f"\n🔬 PHASE 2: Model Comparison")
    logger.info("-" * 40)
    results = compare_models(db_manager, config_dict)
    
    logger.info(f"\n🎯 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Revolutionary quantum-enhanced AI framework demonstrated!")
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum-AI Drug Discovery Framework - KK&GDevOps Professional Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline (complete demonstration)
    python main.py full-pipeline --dataset BBBP --sample-size 1000 --quick
    
    # Ingest data only
    python main.py ingest --dataset BBBP --sample-size 500
    
    # Train single model
    python main.py train --model-type quantum_hybrid --dataset BBBP
    
    # Compare models
    python main.py compare --dataset BBBP --models fingerprint_classifier classical_gnn quantum_hybrid

Built by KK&GDevOps LLC in 40 minutes as a $300K position demonstration.
        """
    )
    
    # Database connection
    parser.add_argument(
        '--database-url',
        default=os.getenv('DATABASE_URL', 
                         'postgresql://quantum_user:quantum_pass_2024@localhost:5432/quantum_drug_discovery'),
        help='PostgreSQL database URL'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full-pipeline', help='Run complete pipeline')
    full_parser.add_argument('--dataset', default='BBBP', help='Dataset name')
    full_parser.add_argument('--sample-size', type=int, default=1000, help='Sample size for demo')
    full_parser.add_argument('--quick', action='store_true', help='Quick mode (reduced epochs)')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest molecular data')
    ingest_parser.add_argument('--dataset', required=True, help='Dataset name')
    ingest_parser.add_argument('--sample-size', type=int, help='Sample size')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train single model')
    train_parser.add_argument('--model-type', required=True, 
                            choices=['fingerprint_classifier', 'classical_gnn', 'quantum_hybrid'],
                            help='Model type to train')
    train_parser.add_argument('--dataset', default='BBBP', help='Dataset name')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--dataset', default='BBBP', help='Dataset name')
    compare_parser.add_argument('--models', nargs='+', 
                              default=['fingerprint_classifier', 'classical_gnn', 'quantum_hybrid'],
                              help='Models to compare')
    compare_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'full-pipeline':
        run_full_pipeline(
            args.database_url, 
            args.dataset, 
            args.sample_size, 
            args.quick
        )
    
    elif args.command == 'ingest':
        db_manager = setup_database(args.database_url)
        ingest_data(db_manager, args.dataset, args.sample_size)
    
    elif args.command == 'train':
        db_manager = setup_database(args.database_url)
        config_dict = {
            'model_type': args.model_type,
            'dataset_name': args.dataset,
            'num_epochs': args.epochs,
            'batch_size': args.batch_size
        }
        train_model(db_manager, config_dict)
    
    elif args.command == 'compare':
        db_manager = setup_database(args.database_url)
        config_dict = {
            'dataset_name': args.dataset,
            'num_epochs': args.epochs
        }
        compare_models(db_manager, config_dict, args.models)


if __name__ == "__main__":
    print("🔬 Quantum-AI Drug Discovery Framework")
    print("🏢 KK&GDevOps LLC Professional Demonstration")
    print("⚛️ Revolutionary quantum computing integration for molecular property prediction")
    print("⏱️  Built in 40 minutes for $300K position demonstration")
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n❌ Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)
