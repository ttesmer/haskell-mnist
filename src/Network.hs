module Network
    ( trainBasic,
      trainDeepNet,
      randomMatrix,
      processMNIST
    ) where

import qualified Data.Matrix as M
import qualified System.Random as R
import qualified Control.Monad as C
import qualified Data.ByteString.Lazy as BS
import qualified Data.Binary as B
import Codec.Compression.GZip (decompress)
import System.Directory (getCurrentDirectory)
import Data.List (genericLength, findIndex)
import Data.Word (Word8)

processMNIST :: IO ()
processMNIST = do
    currentDir <- getCurrentDirectory
    imgs <- decompress <$> BS.readFile (currentDir <> "/data/mnist_dataset/train-images-idx3-ubyte.gz")
    labels <- decompress <$> BS.readFile (currentDir <> "/data/mnist_dataset/train-labels-idx1-ubyte.gz")
    n <- (`mod` 60000) <$> R.randomIO

    putStr . unlines $ [(\x -> renderPixel $ BS.index imgs (n*28^2 + 16 + r*28 + x)) <$> [0..27] | r <- [0..27]] -- render image
    let label = fromInteger $ toInteger $ BS.index labels (n + 8)
    putStrLn $ "label: " ++ show label
    let y = labelToVector label
    putStrLn "correct output:"
    print y
    -- FORWARD
    putStrLn "predicted output:"
    let image = [(\x -> normalize . fromInteger $ toInteger (BS.index imgs (n*28^2 + 16 + r*28 + x))) <$> [0..27] | r <- [0..27]]
    let x = M.fromList 784 1 $ concat image
    syn0 <- randomMatrix 30 784 -- 30x784 matrix
    syn1 <- randomMatrix 10 30 -- 10x30 matrix
    forward x syn0 syn1
      where 
        renderPixel n = let s = " .:oO@" in s !! (((fromIntegral n) * 6) `div` 256)
        printImage i = sequence_ $ print <$> i -- print image greyscale values for debugging

{- very important rules (https://ml-cheatsheet.readthedocs.io/en/latest/linear_algebra.html#matrix-multiplication):
    1. The number of columns of the 1st matrix must equal the number of rows of the 2nd
    2. The product of an M x N matrix and an N x K matrix is an M x K matrix. 
       The new matrix takes the rows of the 1st and columns of the 2nd -}
forward :: M.Matrix Float -> M.Matrix Float -> M.Matrix Float -> IO ()
forward x weightsL0 weightsL1 = do
    let l0 = x -- 784 matrix
    let l1 = sigmoid <$> M.multStd2 weightsL0 l0 -- 30x784*784x1 = 30x1
    let l2 = sigmoid <$> M.multStd2 weightsL1 l1  -- 10x30*30x1 = 10x1, which is the output vector
    print l2
    where
      sigmoid  x = 1 / (1 + (exp (-x)))

normalize :: Fractional a => a -> a
normalize x = x / 255

labelToVector :: Num a => Int -> M.Matrix a
labelToVector l = M.fromList 10 1 $ (fst xs) ++ [1] ++ (snd xs)
  where xs = (splitAt l $ replicate 9 0)

sampleGauss x = (1 / (sqrt $ 2*pi)) * exp (-x^2 / 2)

{----------------------------------
MNIST ABOVE, MINIMAL NET BELOW
----------------------------------}

randomMatrix :: Int -> Int -> IO (M.Matrix Float)
randomMatrix nrows ncols = do 
    randomWeightsList <- C.replicateM (ncols*nrows) $ (R.randomRIO (0 :: Float, 1 :: Float))
    let randomMatrixMeanZero = M.fromList nrows ncols ((\y -> 2*y-1) <$> randomWeightsList)
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
    let l1 = sigmoid <$> M.multStd2 l0 weightsL0 -- 4x3*3x4 = 4x4 matrix
    let l2 = sigmoid <$> M.multStd2 l1 weightsL1 -- 4x4*4x1 = 4x1 matrix

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
