  
    def test(self, x, y):
        

        for idx in range(x.shape[0]):
            
            if idx % x.shape[0]/100 ==0:
                
                print(self.loss)


                net = Model()
net.test(x_test, y_test)