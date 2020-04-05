
class Parameters:
    def __init__(self):
        pass

    def _get_init_pars(self):
        return {
            'model_type': None,
            'model_name_dict':{
                'lr': 'linear regression',
                'blr': 'bayesian linear regression',
                'LR': 'logistic regression',
            },
            'regulation': False,
            'bias': False,
            'binary_label': False,
            'yhat_threshold': None,
            'reg_lambda': None,
            'bay_lambda': None,
            'log': None,
        }

    def get_pars(self, type):
        type = type.split('_')
        if len(type) >= 2:
            model_type, other_pars = type[0], type[1:]
        else:
            model_type, other_pars = type[0], []

        pars = self._get_init_pars()
        pars['model_type'] = model_type
        log = ''
        log += pars['model_name_dict'][model_type]
        reg = bias = binary_label = False
        for par in other_pars:
            if par == 'reg':
                pars['regulation'] = True
                pars['reg_lambda'] = 1.0
                reg = True
            elif par == 'bias':
                pars['bias'] = True
                pars['bay_lambda'] = 1.0
                bias = True
            elif par == 'binary':
                pars['binary_label'] = True
                pars['yhat_threshold'] = 0.5
                binary_label = True
        if reg:
            log += ' /regulation'

        if bias:
            log += ' /bias'

        if binary_label:
            log += ' /binary_label'

        pars['log'] = log


        return pars
