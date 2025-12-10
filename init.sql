-- PostgreSQL Database Schema for Quantum-AI Drug Discovery
-- Production-grade schema with proper indexing and constraints

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Datasets table
CREATE TABLE datasets (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    task_type VARCHAR(50) NOT NULL CHECK (task_type IN ('classification', 'regression')),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Molecules table
CREATE TABLE molecules (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    smiles TEXT UNIQUE NOT NULL,
    canonical_smiles TEXT,
    inchi TEXT,
    inchi_key VARCHAR(255),
    molecular_weight DECIMAL,
    logp DECIMAL,
    num_atoms INTEGER,
    num_bonds INTEGER,
    num_rings INTEGER,
    tpsa DECIMAL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Molecular fingerprints table
CREATE TABLE molecular_fingerprints (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    molecule_id UUID REFERENCES molecules(id) ON DELETE CASCADE,
    fingerprint_type VARCHAR(50) NOT NULL,
    fingerprint_bits BYTEA,
    bit_length INTEGER,
    radius INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(molecule_id, fingerprint_type, radius)
);

-- Bioactivities table
CREATE TABLE bioactivities (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    molecule_id UUID REFERENCES molecules(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    label_value DECIMAL,
    label_binary INTEGER,
    activity_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(molecule_id, dataset_id)
);

-- Training runs table
CREATE TABLE training_runs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    dataset_id UUID REFERENCES datasets(id),
    hyperparameters TEXT,
    status VARCHAR(50) DEFAULT 'running',
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model metrics table
CREATE TABLE model_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    training_run_id UUID REFERENCES training_runs(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    split_type VARCHAR(20) NOT NULL CHECK (split_type IN ('train', 'val', 'test')),
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_molecules_smiles ON molecules(smiles);
CREATE INDEX idx_molecules_canonical_smiles ON molecules(canonical_smiles);
CREATE INDEX idx_molecules_inchi_key ON molecules(inchi_key);
CREATE INDEX idx_molecular_fingerprints_molecule_id ON molecular_fingerprints(molecule_id);
CREATE INDEX idx_molecular_fingerprints_type ON molecular_fingerprints(fingerprint_type);
CREATE INDEX idx_bioactivities_molecule_id ON bioactivities(molecule_id);
CREATE INDEX idx_bioactivities_dataset_id ON bioactivities(dataset_id);
CREATE INDEX idx_training_runs_status ON training_runs(status);
CREATE INDEX idx_model_metrics_run_id ON model_metrics(training_run_id);

-- Insert default datasets
INSERT INTO datasets (name, description, task_type) VALUES
('BBBP', 'Blood-Brain Barrier Penetration', 'classification'),
('Tox21', 'Toxicity in 21st Century', 'classification'),
('ESOL', 'Estimated SOLubility', 'regression'),
('FreeSolv', 'Free Solvation Database', 'regression');

