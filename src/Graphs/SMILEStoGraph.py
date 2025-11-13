from rdkit import Chem
from .Graphs import molecule_graph
import pandas as pd

def SMILEStoGraph(SMILES_str:str):
    
    # Internal Graph logic:
    Atom_values = {
        "C":[1], 
        "O":[2], 
        "N":[3],
        "S":[4],
        "Cl":[5],
        "F":[6],
        }
    
    Bond_values = {
        "SINGLE":[1],
        "DOUBLE":[2],
        "TRIPLE":[3],
        "AROMATIC":[4],
    }

#  Bond Types defined by RDKit:    
#     BondType {
#   UNSPECIFIED = 0 , SINGLE , DOUBLE , TRIPLE ,
#   QUADRUPLE , QUINTUPLE , HEXTUPLE , ONEANDAHALF ,
#   TWOANDAHALF , THREEANDAHALF , FOURANDAHALF , FIVEANDAHALF ,
#   AROMATIC , IONIC , HYDROGEN , THREECENTER ,
#   DATIVEONE , DATIVE , DATIVEL , DATIVER ,
#   OTHER , ZERO
#     }

    vert_vals = []
    edge_vals = []
    
    mol = Chem.MolFromSmiles(SMILES_str)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {SMILES_str}")
    
    # print(mol)
    
    vert_vals = [Atom_values[atom.GetSymbol()] for atom in mol.GetAtoms()]
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    adj = set()
    edge_vals = []
    bonds = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adj.add(frozenset({start+1,end+1}))
        edge_vals.append(Bond_values[str(bond.GetBondType())])
        bonds.append(str(bond.GetBondType()))
        
    # Get Molecule data from COSMO_database
    solvents = pd.read_csv("/home/stefan/GIT_Repositories/GraphNeuralNetworks/Solvents.txt")
    if SMILES_str not in solvents["SMILES"].values:
        # raise ValueError(f"SMILES string {SMILES_str} not found in the solvents database.")
        print(f"SMILES string {SMILES_str} not found in the solvents database.")
        BP = None
        MP = None
        COMSO_name = None
    else:
        BP = solvents[solvents["SMILES"] == SMILES_str]["BP"].values[0].item()
        MP = solvents[solvents["SMILES"] == SMILES_str]["MP"].values[0].item()
        COMSO_name = solvents[solvents["SMILES"] == SMILES_str]["COSMO_name"].values[0]
    global_vals = {
        "BP": BP,
        "MP": MP
    }
    
    graph = molecule_graph(
        global_vals=global_vals,
        vertex_vals=vert_vals,
        edge_vals=edge_vals,
        adj=adj,
        SMILES = SMILES_str,
        COSMO_name = COMSO_name,
        atoms = atoms,
        bonds = bonds
    )
    
    return graph 