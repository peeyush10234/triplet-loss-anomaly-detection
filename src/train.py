import model
import loss_func
import data_loader
class train_val():
    def __init__(self, df, train_loader, val_loader, val_split=0.2):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_loader_obj = data_loader(df)
        self.val_split = val_split
        self.df_train, self.df_val = self.data_loader_obj.create_train_val_split(val_split)
        self.train_loader = self.
    def train_func()