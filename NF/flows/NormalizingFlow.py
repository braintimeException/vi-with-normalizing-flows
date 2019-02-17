from NF.flows.SylvesterTranformation import SylvesterTransformation, TriagSylvesterTransformation

class NormalizingFlow:
    
    """Normalizing flow which comprises several Transformation's of the same type"""

    def __init__( self, transformation, dim:int, K:int, num_hidden:int=20, transformations=None ):
        """Init

        Args:
        transformation -- class of the transformation to be used in this flow
        dim            -- dimension of z
        K              -- flow length (=number of chained Transformation's)

        Kwargs:
        num_hidden      -- number of hidden units, SylvesterTranformation only
        transformations -- list with transformations. If provided, these 
                            transformations will be used instead of generating new
        """

        self.K = K
        self.dim = dim
        
        if transformations is None:
            if issubclass(transformation, SylvesterTransformation):
                if issubclass(transformation, TriagSylvesterTransformation):
                    transformations = [ transformation( dim, num_hidden, i%2==0 ) for i in range( K ) ]
                else:
                    transformations = [ transformation( dim, num_hidden ) for i in range( K ) ]
            else:
                transformations = [ transformation( dim ) for i in range( K ) ]
        self.flow = transformations
        self.nParams = self.flow[0].get_num_params()
        
    def get_last_log_det(self):
        """Get log determinant of the last Transformation in the flow"""
        return self.flow[-1].log_det
    
    def get_sum_log_det(self):
        """Get summed log jacobians of all Transformation's in the flow"""
        ret = 0
        for trans in self.flow:
            ret += trans.log_det
        return ret
        
    def forward( self, z, params ):
        """Pass z through all Transformation's in the flow
        
        Args:
        z       -- variable which will be transformed
        params  -- parameters for this flow

        Returns transformed z' of the same shape as z
        """
        for i, transf in enumerate( self.flow ):
            z = transf.forward(z, params[i])
        return z
