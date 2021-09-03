module Lib
    ( trainBasic,
      trainDeepNet,
      randomMatrix,
      processMNIST
    ) where

import qualified Data.Matrix as M
import qualified System.Random as R
import qualified Control.Monad as C
import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import System.Directory (getCurrentDirectory)
import Data.List (genericLength)
import GHC.Word (Word8)

processMNIST :: IO ()
processMNIST = do
    currentDir <- getCurrentDirectory
    imgs <- decompress <$> BS.readFile (currentDir <> "/data/mnist_dataset/train-images-idx3-ubyte.gz")
    labels <- decompress <$> BS.readFile (currentDir <> "/data/mnist_dataset/train-labels-idx1-ubyte.gz")
    n <- (`mod` 60000) <$> R.randomIO

    -- render image
    putStr . unlines $ [(\x -> renderPixel $ BS.index imgs (n*28^2 + 16 + r*28 + x)) <$> [0..27] | r <- [0..27]]
    -- print label for image
    print $ BS.index labels (n + 8)
    -- print greyscale/pixel values
    sequence_ $ print <$> [(\x -> BS.index imgs (n*28^2 + 16 + r*28 + x)) <$> [0..27] | r <- [0..27]]
    print [sampleGauss x | x <- [1..100]]
      where 
          renderPixel n = let s = " .:oO@" in s !! (((fromIntegral n) * 6) `div` 256)

sampleGauss x = (1 / (sqrt $ 2*pi)) * exp (-x^2 / 2)

{----------------------------------
MNIST ABOVE, MINIMAL NET BELOW
----------------------------------}

randomMatrix :: Int -> Int -> IO (M.Matrix Float)
randomMatrix ncols nrows = do 
    randomWeightsList <- C.replicateM (ncols*nrows) $ (R.randomRIO (0 :: Float, 1 :: Float))
    let randomMatrixMeanZero = M.fromList ncols nrows ((\y -> 2*y-1) <$> randomWeightsList)
    return randomMatrixMeanZero

trainBasic :: Integer -> M.Matrix Float -> M.Matrix Float -> M.Matrix Float -> IO ()
trainBasic epoch x y weights = do
    let l1 = sigmoid <$> M.multStd2 x weights
    let l1Error = y - l1
    let l1Delta = M.elementwise (*) l1Error $ sigmoid' <$> l1
    let updatedWeights = weights + M.multStd2 (M.transpose x) l1Delta
    if epoch == 0
      then print l1
      else trainBasic (epoch-1) x y updatedWeights
       where sigmoid  x = 1 / (1 + (exp (-x)))
             sigmoid' x = x * (1 - x)

trainDeepNet :: Integer -> M.Matrix Float -> M.Matrix Float -> M.Matrix Float -> M.Matrix Float -> IO ()
trainDeepNet epoch x y weightsL0 weightsL1 = do

    let l0 = x
    let l1 = sigmoid <$> M.multStd2 l0 weightsL0
    let l2 = sigmoid <$> M.multStd2 l1 weightsL1

    let l2Error = y - l2

    -- line below is not ideal, should be revisited later
    C.when (epoch `mod` 10000 == 0) (putStrLn $ "Error: " <> (show $ arithmeticMean (M.toList $ abs l2Error)))

    let l2Delta = M.elementwise (*) l2Error $ sigmoid' <$> l2
    let l1Error = M.multStd2 l2Delta (M.transpose weightsL1)
    let l1Delta = M.elementwise (*) l1Error $ sigmoid' <$> l1

    let updatedWeightsL1 = weightsL1 + M.multStd2 (M.transpose l1) l2Delta
    -- putStrLn $ "wl1\n" ++ (show updatedWeightsL1)
    let updatedWeightsL0 = weightsL0 + M.multStd2 (M.transpose l0) l1Delta
    -- putStrLn $ "wl0\n" ++ (show updatedWeightsL0)

    if epoch /= 0 
        then trainDeepNet (epoch-1) x y updatedWeightsL0 updatedWeightsL1
        else print l1 >> print l2
      where sigmoid  x = 1 / (1 + (exp (-x)))
            sigmoid' x = x * (1 - x)

arithmeticMean :: [Float] -> Float
arithmeticMean xs = (sum xs) / (genericLength $ xs)
