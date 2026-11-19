import copy


class _ConfigDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


try:
    import ml_collections as mlc
    _config_dict = mlc.ConfigDict
except ModuleNotFoundError:
    _config_dict = _ConfigDict


config = _config_dict(
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
                # These five hold the ACTIVE --label_metric, not RMSD: lDDT when
                # label_metric is an lDDT, RMSD when it is an RMSD. Named for the
                # role, not the unit, so the name stays true either way. The epoch
                # log records which metric was active in its '# label_metric=' line.
                'metric_diff' :0.0,
                #'pLDDT'    :0.0,
                'top1_metric' :0.0,
                'top3_metric' :0.0,
                'top5_metric' :0.0,
                'top10_metric':0.0,
                      }
                }
    }
)
