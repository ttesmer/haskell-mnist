{-# LANGUAGE BangPatterns #-}
module Main where
import           Control.Monad
import           Network
import           Numeric.LinearAlgebra
import           System.Directory      (getCurrentDirectory)
import           System.IO

netConfig :: IO NeuralNet
netConfig = initWB NN
    { weights   = []
    , biases    = []
    , eta       = 3.0
    , epochs    = 30
    , layers    = 2
    , layerSize = 30
    , batchSize = 10  -- has to be a divisor of 50000
    , trainData = []
    , testData  = []
    }



main :: IO ()
main = do
    dir <- getCurrentDirectory

    trainingData <- loadData
    testingData <- loadTestData
    net <- netConfig
    model <- train net {trainData = trainingData, testData = testingData}

    forM_ (enum $ weights model) $ \(i, m) -> do
        let path = (dir ++ "/data/model/w" ++ (show i) ++ ".txt") :: FilePath
        saveMatrix path "%g" m
        return ()
    forM_ (enum $ biases model) $ \(i, b) -> do
        let path = (dir ++ "/data/model/b" ++ (show i) ++ ".txt") :: FilePath
        saveMatrix path "%g" $ asColumn b
        return () 
    where
      enum :: [a] -> [(Int, a)]
      enum a = zip (enumFrom 0) a
