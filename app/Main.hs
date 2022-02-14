{-# LANGUAGE BangPatterns #-}
module Main where
import           Control.Monad
import           Data.List
import           Numeric.LinearAlgebra (asColumn, saveMatrix)
import           System.Directory      (getCurrentDirectory)
import           System.Environment
import           System.Exit
import           System.IO
import           Text.Printf

import           ConvNet               as Conv
import           Helper
import           Network               as MLP

main :: IO ()
main = do
    argv <- getArgs

    (trainData, validationData) <- loadData
    testData <- loadTestData

    --  | TODO: add --CUDA option (Accelerate)
    model <- case argv of
                "-MLP":_ -> do
                    net <- MLP.initWB NN
                        { weights       = []
                        , biases        = []
                        , eta           = 0.1
                        , lambda        = 5.0
                        , totalEpochs   = 30
                        , epochs        = 30
                        , layers        = 2
                        , layerSize     = 100
                        , MLP.batchSize = 10
                        , MLP.trainData = trainData
                        , testData      = testData }
                    printf ("Running multilayer perceptron "
                        <> "with cross-entropy, mini-batch SGD, "
                        <> "η of %.2f, λ of %.2f and %d neurons:\n")
                            (MLP.eta net) (MLP.lambda net) (MLP.layerSize net)
                    model <- MLP.train net
                    return model
                "-CNN":_ -> do
                    net <- Conv.initKernels $
                        Conv.CNN
                            { kernels        = []
                            , kernelSize     = (5, 5)
                            , nkernels       = 1
                            , Conv.batchSize = 10
                            , Conv.trainData = squareImgMats trainData}
                    model <- return $ Conv.train net
                    print model >> exitSuccess
                _ -> usage >> exitSuccess

    case find (isPrefixOf "--") argv of
        Just "--save-result" -> do
            dir  <- getCurrentDirectory
            forM_ (enum $ weights model) $ \(i, m) -> do
                let path = (dir <> "/data/result/w" <> (show i)) :: FilePath
                saveMatrix path "%g" m
                return ()
            forM_ (enum $ biases model) $ \(i, b) -> do
                let path = (dir <> "/data/result/b" <> (show i)) :: FilePath
                saveMatrix path "%g" $ asColumn b
                return ()
            printf "Done, saved result in %s\n" (dir <> "/data/result/")
            where
              enum :: [a] -> [(Int, a)]
              enum a = zip (enumFrom 0) a
        Nothing -> putStrLn "Done"

usage :: IO ()
usage = putStrLn $
    ("Usage:\n\n$ stack exec -- hmnist-exe [-abc] [--xyz]\nor\n"
        <> "$ cabal v2-exec -- ``\n\n"
        <> "Available flags:\n"
        <> "-CNN          # Convolutional Network\n"
        <> "-MLP          # Normal ANN (multilayer perceptron)\n"
        <> "--save-result # Save weights and biases")

