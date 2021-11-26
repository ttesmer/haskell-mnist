{-# LANGUAGE BangPatterns   #-}
{-# LANGUAGE NamedFieldPuns #-}
module Network
    ( NeuralNet (..),
      train,
      test,
      initWB,
      loadData,
      loadTestData,
    ) where

import           Codec.Compression.GZip (decompress)
import           Control.Monad
import qualified Data.Array.IO          as A
import qualified Data.ByteString        as BS
import qualified Data.ByteString.Lazy   as BL
import           Data.Int               (Int64)
import           Data.List
import           GHC.Base               (build)
import           Numeric.LinearAlgebra  hiding (build, normalize)
import           Prelude                hiding ((<>))
import           System.Directory       (getCurrentDirectory)
import qualified System.Random          as R
import           Util

type Image        = Vector Double
type Label        = Vector Double
type Bias         = Vector Double
type Weight       = Matrix Double
type Activation   = Matrix Double
type Nabla        = Matrix Double
type ImagesMat    = Matrix Double
type LabelsMat    = Matrix Double

data NeuralNet = NN
    { weights   :: ![Weight]   -- ^ List of all weights
    , biases    :: ![Bias]    -- ^ List of all biases
    , eta       :: !Double    -- ^ Learning rate
    , epochs    :: !Int       -- ^ No. of epochs
    , layers    :: !Int       -- ^ No. of layers
    , layerSize :: !Int       -- ^ Size of hidden layers
    , batchSize :: !Int       -- ^ Size of mini-batch
    , trainData :: ![(Image, Label)]
    , testData  :: ![(ImagesMat, Int)]
    } deriving (Show, Eq)

test :: NeuralNet -> IO ()
test net@NN{epochs, biases, weights, testData} = do
    let (imgs, labels) = unzip testData
    let guesses = [guess img (zip weights $ toMatrix <$> biases) | img <- imgs]
    let c = sum $ (\(a,b) -> if a == b then 1 else 0) <$> zip guesses labels
    putStrLn $ "Epoch #" ++ (show $ 30-epochs) ++ ": " ++ (show $ c/100) ++ "%"
  where
    guess :: ImagesMat -> [(Weight, Matrix Double)] -> Int
    guess i wb = maxIndex . flatten . head $ feedforward i wb

-- | Fully matrix-based approach to backpropagation.
train :: NeuralNet -> IO NeuralNet
train net@NN{weights, batchSize, epochs, trainData} = do
    shuffledData <- shuffle trainData
    let miniBatches = chunkList batchSize shuffledData
    let batches = fmap getMiniBatch miniBatches
    let !(newWs, newBs) = (recurTest batchSize (weights, biases net) batches) :: ([Weight], [Bias])
    let newNet = net { weights = newWs, biases = newBs, epochs = epochs-1 }
    test newNet
    case epochs of
        1 -> return newNet
        _ -> train newNet

recurTest :: Int -> ([Weight], [Bias]) -> [(ImagesMat, LabelsMat)] -> ([Weight], [Bias])
recurTest batchSize !wb [batch] = trainBatch batchSize wb batch
recurTest batchSize !wb (firstBatch:batches) = recurTest batchSize (trainBatch batchSize wb firstBatch) batches

trainBatch :: Int -> ([Weight], [Bias]) -> (ImagesMat, LabelsMat) -> ([Weight], [Bias])
trainBatch batchSize !(ws, bs) (mX, mY) = backprop mY ws bs $ feedforward mX (zip ws $ bTm bs batchSize)

-- | Make biases suitable for full-matrix backprop.
bTm :: [Bias] -> Int -> [Matrix Double]
bTm biases batchSize = fmap (fromColumns . replicate batchSize) biases

-- | Turn mini-batch matrices into one matrix.
getMiniBatch :: [(Image, Label)] -> (ImagesMat, LabelsMat)
getMiniBatch mBatchData = (fromColumns imgs, fromColumns labels)
  where
    (imgs, labels) = unzip mBatchData

feedforward :: ImagesMat -> [(Weight, Matrix Double)] -> [Activation]
feedforward = scanr (\(w, b) a -> sigmoid $ w <> a + b)

backprop :: LabelsMat -> [Weight] -> [Bias] -> [Activation] -> ([Weight], [Bias])
backprop y ws bs (aL:as) = (zipWithSafe (-) ws (zipWithSafe (<>) deltaList (tr <$> as)), zipWithSafe (-) bs $ meanRows <$> deltaList)
    where
        deltaList :: [Matrix Double]
        deltaList = init $ deltas [edgeDelta y aL] y as ws

deltas :: [Matrix Double]
       -> LabelsMat
       -> [Activation]
       -> [Weight]
       -> [Matrix Double]
deltas deltal _ [] [] = deltal
deltas deltal y (a:as) (wlp1:ws) = deltas (deltal++[(tr wlp1 <> last deltal) * sigmoid' a]) y as ws

edgeDelta :: LabelsMat -- ^ Correct output Y for input X
           -> Activation -- ^ Predicted ouput for input X
           -> Matrix Double
edgeDelta y aL = (aL - y) * sigmoid' aL

{-- Arithmetic Functions --}
cost :: Matrix Double -> Matrix Double -> Matrix Double
cost a y = ((a-y)**2) / 2

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + (exp (-x)))

sigmoid' :: Floating a => a -> a
sigmoid' x = x * (1 - x)

normalize :: (Integral a, Floating b) => a -> b
normalize x = (fromIntegral x) / 255

gauss :: IO Double
gauss = do
    u1 <- R.randomRIO (0 :: Double, 1 :: Double)
    u2 <- R.randomRIO (0 :: Double, 1 :: Double)
    return $ (sqrt (-2*log u1)) * (cos (2*pi*u2))

{-- Data Processing Functions --}
initWB :: NeuralNet -> IO NeuralNet
initWB net@NN{batchSize = bs, layerSize = ls, layers = n} = do
    headW <- randn ls 784
    midW  <- replicateM (n-2) (randn ls ls)
    lastW <- randn 10 ls

    initB <- replicateM (n-1) (randn ls 1)
    lastB <- randn 10 1

    let weights = reverse $ headW : midW ++ [lastW]
    let biases = reverse . fmap flatten $ initB ++ [lastB]

    return net { weights = weights
               , biases  = biases }

randomMatrix :: Int -> Int -> IO (Matrix Double)
randomMatrix nrows ncols = do
    gaussList <- replicateM (ncols*nrows) gauss
    return $ (nrows><ncols) gaussList

loadData :: IO [(Vector Double, Vector Double)]
loadData = do
    trainImgs <- getData "train-images-idx3-ubyte.gz"
    trainLabels <- getData "train-labels-idx1-ubyte.gz"
    let !labels = BL.toStrict trainLabels
    let !imgs = BL.toStrict trainImgs
    let !l = [vectorizeLabel $ getLabel n labels | n <- [0..49999]] :: [Label]
    let !i = [getImage n imgs | n <- [0..49999]] :: [Image]
    return $ zip i l

loadTestData :: IO [(Matrix Double, Int)]
loadTestData = do
    testImgs <- getData "t10k-images-idx3-ubyte.gz"
    testLabels <- getData "t10k-labels-idx1-ubyte.gz"
    let !labels = BL.toStrict testLabels
    let !imgs = BL.toStrict testImgs
    let !l = [getLabel n labels | n <- [0..9999]] :: [Int]
    let !i = asColumn <$> [getImage n imgs | n <- [0..9999]] :: [Matrix Double]
    return $ zip i l

getData :: FilePath -> IO BL.ByteString
getData path = do
    currentDir <- getCurrentDirectory
    fileData <- decompress <$> BL.readFile (currentDir ++ "/data/mnist_dataset/" ++ path)
    return fileData

getImage :: Int -> BS.ByteString -> Image
getImage n imgs = fromList [normalize $ BS.index imgs (16 + n*784 + s) | s <- [0..783]]

-- | `getImage` but lazy
getImage' :: Int64 -> BL.ByteString -> Image
getImage' n imgs = fromList [normalize $ BL.index imgs (16 + n*784 + s) | s <- [0..783]]

getLabel :: Num a => Int -> BS.ByteString -> a
getLabel n labels = fromIntegral $ BS.index labels (n+8)

getGuess :: [Activation] -> Int
getGuess = maxIndex . flatten . head

vectorizeLabel :: Int -> Vector Double
vectorizeLabel l = fromList $ x ++ 1 : y
    where (x,y) = splitAt l $ replicate 9 0

{-- HELPER FUNCTIONS --}
toMatrix :: Vector Double -> Matrix Double
toMatrix = reshape 1

-- | Means of rows in matrix as a column vector
meanRows :: Matrix Double -> Vector Double
meanRows = fst . meanCov . tr

-- | zipWith, but throws error if lengths don't match
zipWithSafe :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWithSafe f as bs
    | length as == length bs = zipWith f as bs
    | otherwise = error $ "zipWith: lengths don't match (" ++ (show $ length as) ++ " " ++ (show $ length bs) ++ ")"

-- | Randomly shuffle a list
--   /O(N)/
shuffle :: [a] -> IO [a]
shuffle xs = do
        ar <- newArray n xs
        forM [1..n] $ \i -> do
            j <- R.randomRIO (i,n)
            vi <- A.readArray ar i
            vj <- A.readArray ar j
            A.writeArray ar j vi
            return vj
  where
    n = length xs
    newArray :: Int -> [a] -> IO (A.IOArray Int a)
    newArray n xs =  A.newListArray (1,n) xs
