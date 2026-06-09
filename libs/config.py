import copy
import ml_collections as mlc
config = mlc.ConfigDict(
    {
        "heads": {
                'aa_score':True,
                'drsa_pred':True,
                'unbound_rsa_pred':False,
                
                },
        "loss": {
            "weight": {
                'final_loss'        :0.0,
                'best_rank'       :0.0,
                #'lddt_diff'      :0.0,
                'rmsd_diff' :0.0,
                #'pLDDT'    :0.0,
                'top1_rmsd' :0.0,
                'top3_rmsd' :0.0,
                'top5_rmsd' :0.0,
                'top10_rmsd':0.0,
                # 'topn_rmsd':{'top1_rmsd':0.0,'top3_rmsd':0.0,'top5_rmsd':0.0,'top10_rmsd':0.0},
                      }
                }
    }
)
