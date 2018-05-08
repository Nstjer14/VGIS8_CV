# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:10:47 2018
http://www.onlamp.com/pub/a/python/2004/12/02/tdd_pyunit.html
http://www.onlamp.com/pub/a/python/2005/02/03/tdd_pyunit2.html
http://pyunit.sourceforge.net/pyunit.html
@author: Shaggy
"""

import unittest
import pandas as pd
import numpy as np
import load_images_2_python
import NSC_miniproject_1 as naiveImplementation
import NSC_miniproject_2 as optimisedImplementation


class miniProjectTestCases(unittest.TestCase):

    def setUp(self):
        self.dataFrame = load_images_2_python.dataFrame

    def tearDown(self):
        del self.dataFrame
                
    
    def test_seq_para_dataframe(self):
        dataFrame_seq = pd.read_pickle("pythonDatabase_seq")
        dataFrame_para = pd.read_pickle("pythonDatabase_para")
        check = np.isclose(dataFrame_para.featureVector.tolist(),dataFrame_seq.featureVector.tolist())
        assert np.all(check),'Dataframes are not equal'
     
    def test_naive_optimised_histogram_output(self):
        testImage = self.dataFrame.image[0]
        #histoFrac = 0.1
        recognitionValue = 10
        naiveOutput = naiveImplementation.equalisehistogram(testImage,recognitionValue)
        optimisedOutput = optimisedImplementation.equalisehistogram(testImage,recognitionValue)
        check = np.isclose(naiveOutput,optimisedOutput)
        self.assertTrue(np.all(check),msg='Histogram outputs are not equal')
        
    def test_naive_optimised_noiseremove_output(self):
        testImage = self.dataFrame.image[0]
        histoFrac = 0.1
        recognitionValue = 10
        naiveOutput = naiveImplementation.noiseremover(testImage,histoFrac,recognitionValue)
        optimisedOutput = optimisedImplementation.noiseremover(testImage,histoFrac,recognitionValue)
        check = np.isclose(naiveOutput,optimisedOutput)
        self.assertTrue(np.all(check),msg='Noise removal and reconstruction outputs are not equal')

    

def main():
    unittest.main()
    
if __name__ == '__main__':
    main()