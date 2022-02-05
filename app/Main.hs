{-# LANGUAGE BangPatterns #-}
module Main where
import           Control.Monad
import           Network
import           Numeric.LinearAlgebra
import           System.Directory      (getCurrentDirectory)
import           System.IO
import           Text.Printf

netConfig :: IO NeuralNet
netConfig = initWB NN
    { weights   = []
    , biases    = []
    , eta       = 0.1
    , lambda    = 5.0
    , epochs    = 60
    , layers    = 2
    , layerSize = 100
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

    printf "Running cross-entropy mini-batch SGD with η of %.2f, λ of %.2f and %d neurons:\n" (eta net) (lambda net) (layerSize net)
    model <- train net {trainData = trainingData, testData = testingData}

    forM_ (enum $ weights model) $ \(i, m) -> do
        let path = (dir ++ "/data/result/w" ++ (show i)) :: FilePath
        saveMatrix path "%g" m
        return ()
    forM_ (enum $ biases model) $ \(i, b) -> do
        let path = (dir ++ "/data/result/b" ++ (show i)) :: FilePath
        saveMatrix path "%g" $ asColumn b
        return ()
    where
      enum :: [a] -> [(Int, a)]
      enum a = zip (enumFrom 0) a
