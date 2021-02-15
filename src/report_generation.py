import pandas as pd
from config import configs
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def format_fig(fig):
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue())
    return encoded

class report_generation():
    def __init__(self, df_train, df_val, loss_dict, val_dict, model_param):
        self.df_train = df_train
        self.df_val = df_val
        self.loss_dict = loss_dict
        self.val_dict = val_dict
        self.model_param = model_param
    
    def create_report_df(self):
        image_path = pd.read_csv('../image_path.csv')
        original_data_dict = {
            'Total Data' : image_path.shape[0],
            'Normal Data' : image_path.loc[image_path['label'] == 'OK'].shape[0],
            'Anomaly Data' : image_path.loc[image_path['label'] == 'NG'].shape[0]
        }

        original_data_df = pd.DataFrame(list(original_data_dict.items()), columns = ['Specs', 'Size'])
        self.original_data_df_path = str(configs.reports_dir / 'original_data_df.csv')
        original_data_df.to_csv(self.original_data_df_path, index=False)

        aug_df = pd.read_csv('../crop_image_paths.csv')
        train_aug_ok = self.df_train.loc[self.df_train['label'] == 'OK'].shape[0]
        train_aug_ng = self.df_train.loc[self.df_train['label'] == 'NG'].shape[0]
        val_aug_ok = self.df_val.loc[self.df_val['label'] == 'OK'].shape[0]
        val_aug_ng = self.df_val.loc[self.df_val['label'] == 'NG'].shape[0]
        augmented_data = [
            [train_aug_ok, train_aug_ng, train_aug_ok+train_aug_ng],
            [val_aug_ok, val_aug_ng, val_aug_ok+val_aug_ng]
        ]

        augmented_data_df = pd.DataFrame(augmented_data, columns= ['Normal', 'Anomaly', 'Total'],
                                         index = {0:'Training', 1:'Validation'})
        self.augmented_data_df_path = str(configs.reports_dir / 'augmented_data_df.csv')
        augmented_data_df.to_csv(self.augmented_data_df_path)

        self.model_param_path = str(configs.reports_dir / 'model_param.csv')
        self.model_param.to_csv(self.model_param_path, index=False)

        metrics_data = [
            [self.val_dict['true_pos'], self.val_dict['false_pos'],
             self.val_dict['true_neg'], self.val_dict['false_neg'],
             self.val_dict['precision'], self.val_dict['recall']]
        ]
        metrics_df = pd.DataFrame(metrics_data, columns=['TP', 'FP', 'TN', 'FN', 'precision', 'recall'])
        self.metrics_df_path = str(configs.reports_dir / 'metrics_df.csv')
        metrics_df.to_csv(self.metrics_df_path, index=False)
    
    def generate_report(self):
        original = pd.read_csv(self.original_data_df_path)\
                     .to_html()\
                     .replace('<table border="1" class="dataframe">','<table class="table table-striped">')
        
        aug_df = pd.read_csv(self.augmented_data_df_path)
        aug_df.rename(index={0:'Training',1:'Validation'}, inplace=True)
        aug_df = aug_df.drop('Unnamed: 0', axis=1)
        aug_df = aug_df.to_html().replace('<table border="1" class="dataframe">','<table class="table table-striped">')

        model_param = pd.read_csv(self.model_param_path)\
                        .to_html()\
                        .replace('<table border="1" class="dataframe">','<table border="1" class="table table-striped">')

        metric = pd.read_csv(self.metrics_df_path)
        conf_mat = pd.DataFrame([[metric['TP'][0], metric['FP'][0]],[metric['FN'][0], metric['TN'][0]]])
        conf_mat.columns = ['Actual positives', 'Actual negatives']
        conf_mat.index = ['Predicted positives', 'Predicted negatives']
        conf_mat = conf_mat\
                    .to_html()\
                    .replace('<table border="1" class="dataframe">','<table border="1" class="table table-striped">')
        
        x = [idx*10 for idx in range(len(self.loss_dict['step_loss_lrc']))]
        fig_recons_loss = plt.figure()
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss Curve')
        plt.plot(x, self.loss_dict['step_loss_lrc'])
        plt.plot(x, self.loss_dict['step_val_loss_lrc'])
        plt.legend(labels=['recons_train_loss', 'recons_val_loss'])

        fig_triplet_loss = plt.figure()
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Triplet Loss Curve')
        plt.plot(x, self.loss_dict['step_loss_lt'])
        plt.plot(x, self.loss_dict['step_val_loss_lt'])
        plt.legend(labels=['triplet_train_loss', 'triplet_val_loss'])

        html_string = '''
<html>
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{ margin:0 100; background:whitesmoke; }
                h1{text-align: center; text-decoration:underline; }
        </style>
    </head>
	<body>
	    <h1> PERFORMANCE REPORT </h1>
            <h2> DATA </h2>
            <h3> Original </h3>'''+original+'''
            <h3> Data Set Distribution (Normal+Augmented Images) </h3>'''+aug_df+'''
	    <h2> MODEL PARAMETERS </h2>'''+model_param+'''
	    <h2> RESULTS </h2>
	        <h3> Recons Loss </h3>'''+"<img src='data:image/png;base64,{}'>".format(format_fig(fig_recons_loss).decode("utf-8"))+'''
	        <h3> Triplet Loss </h3>'''+"<img src='data:image/png;base64,{}'>".format(format_fig(fig_triplet_loss).decode("utf-8"))+'''
	        <h3> Confusion Matrix </h3>'''+conf_mat+'''
                <h3> Precision: {0:.3f} &percnt;'''.format(metric['precision'][0]*100)+''' </h3>
                <h3> Recall: {0:.3f} &percnt;'''.format(metric['recall'][0]*100)+''' </h3>
        </body>
</html>
'''
        html_report_path = str(configs.reports_dir / 'project_report.html')
        with open(html_report_path,'w') as f:
            f.write(html_string)



