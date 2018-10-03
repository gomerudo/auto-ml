
class DataLoader :

  def __init__(self, dataset) :
    self.dataset = dataset
  
  def loadDataset(self) :
    if self.dataset : 
      return True
      # raise ValueError("filepath cannot be null")
    else :
      return False