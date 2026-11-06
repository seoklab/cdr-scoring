MAX_REL_INDEX=32

DECOY_NUM_FALC=1000
DECOY_NUM_PERTMD=32
DECOY_NUM_COMMAT=32
DECOY_NUM_IGFOLD=1
DECOY_NUM_64=64
DECOY_NUM_256=256

AA3_IDX={'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,'GLU':5,'GLN':6,'GLY':7,'HIS':8,'ILE':9,\
    'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,'SER':15,'THR':16,'TRP':17,'TYR':18,'VAL':19,'UNK':20}
AA1_IDX='ARNDCEQGHILKMFPSTWYVX'
REF_CHAIN='ABCDEFGHIJKLMNOPQRSTQUVWXYZabcdefghijklmnopqrstuvwxyz'

ATOM_TYPES = [
    'NONE', 'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 
    'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2',
    'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 
    'CZ2', 'CZ3', 'NZ'#, 'OXT'
] # OXT is determined by the other atoms. so excluded.

ATOM_ORDER = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES)}
CA_IDX = ATOM_ORDER['CA']
CB_IDX = ATOM_ORDER['CB']

restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
}
ALL_NUM_ATOM = len(ATOM_TYPES) # := 36
MAX_NUM_ATOM = len(restype_name_to_atom14_names['TRP']) # := 14

# ── atom14 → ATOM_TYPES lookup (side-chain atom encoder) ──────────────────────
# Row = AA3_IDX[resname], column = atom14 slot (0 N, 1 CA, 2 C, 3 O, 4 CB, 5.. side
# chain). Value = index into ATOM_TYPES; 0 ('NONE') marks an unused slot. Derived
# from restype_name_to_atom14_names so the two never drift apart.
ATOM14_TYPE_IDX = [[0] * MAX_NUM_ATOM for _ in range(len(AA3_IDX))]
for _resname, _names in restype_name_to_atom14_names.items():
    _row = ATOM14_TYPE_IDX[AA3_IDX[_resname]]
    for _slot, _name in enumerate(_names):
        _row[_slot] = ATOM_ORDER.get(_name, 0) if _name else 0
del _resname, _names, _row, _slot, _name

# atom14 slots 0-3 are the backbone (N, CA, C, O); 4+ are CB and beyond.
ATOM14_BACKBONE_SLOTS = 4
