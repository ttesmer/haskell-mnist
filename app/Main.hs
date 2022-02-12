{-# LANGUAGE BangPatterns #-}
module Main where
import           Control.Monad
import           Numeric.LinearAlgebra
import           System.Directory      (getCurrentDirectory)
import           System.IO
import           Text.Printf

import           Network
import           Helper

-- | Tune hyperparameters 
-- | of new NeuralNet below.
-- | `batchSize` has to be a
-- | divisor of 50000
netConfig :: IO NeuralNet
netConfig = initWB NN
    { weights     = []
    , biases      = []
    , eta         = 0.1
    , lambda      = 5.0
    , totalEpochs = 30
    , epochs      = 30
    , layers      = 2
    , layerSize   = 100
    , batchSize   = 10
    , trainData   = []
    , testData    = []
    }

main :: IO ()
main = do
    dir <- getCurrentDirectory

    (trainData, validationData)  <- loadData
    testData <- loadTestData
    net <- netConfig

    printf "Running cross-entropy mini-batch SGD with η of %.2f, λ of %.2f and %d neurons:\n" (eta net) (lambda net) (layerSize net)

    -- train the model and test using testData or validationData
    model <- train net {trainData = trainData, testData = testData}

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
