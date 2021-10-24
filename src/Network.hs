{-# LANGUAGE NamedFieldPuns #-}
module Network
    ( NeuralNet (..),
      train,
      initWB,
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

type Activation   = Matrix Double
type Weight       = Matrix Double
type Bias         = Matrix Double
type Image        = Matrix Double
type Label        = Matrix Double
type ImagesMat    = Matrix Double
type LabelsMat    = Matrix Double

data NeuralNet = NN
    { weights   :: ![Weight]  -- ^ List of all weights
    , biases    :: ![Bias]    -- ^ List of all biases
    , eta       :: !Double    -- ^ Learning rate
    , epochs    :: !Int       -- ^ No. of epochs
    , layers    :: !Int       -- ^ No. of layers
    , layerSize :: !Int       -- ^ Size of hidden layers
    , batchSize :: !Int       -- ^ Size of mini-batch
    } deriving (Show, Eq)

test :: [(Weight, Bias)] -> IO String
test learnedWB = do
    (imgs, labels) <- loadTestData
    let preds = [ getGuess . head $ feedforward img learnedWB | img <- imgs]
    let correct = sum $ (\(a,b) -> if a == b then 1 else 0) <$> zip preds labels
    return $ (show correct) ++ "/" ++ (show 10000) ++ " (" ++ (show ((correct/10000)*100 )) ++  "%)"

-- | Fully matrix-based approach to backpropagation.
train :: NeuralNet -> IO NeuralNet
train net@NN{weights, biases, batchSize, epochs} = do
    (mBatchData, restData) <- return . splitAt batchSize =<< shuffle =<< loadData
    let (miniBatchX, miniBatchY) = getMiniBatch mBatchData
    let activations = feedforward miniBatchX (zip weights biases)
    -- let newNet = backprop miniBatchY weights activations
    print epochs
    case epochs of
        0 -> return net
        _ -> train net { epochs = epochs-1 }

-- | Turn mini-batch matrices into one matrix.
getMiniBatch :: [(Image, Label)] -> (ImagesMat, LabelsMat)
getMiniBatch mBatchData = (concatMat imgs, concatMat labels)
  where
    (imgs, labels) = unzip mBatchData
    concatMat = foldl1 (|||)

feedforward :: Image -> [(Weight, Bias)] -> [Activation]
feedforward = scanr (\(w, b) a -> sigmoid $ w <> a + b)

backprop :: Label -> [Weight] -> [Activation] -> [Matrix Double]
backprop y weights (aL:as) = unpackTuples $ drop 1 [(nw, nb) | (dn, nw, nb) <- scanl' delta (edgeDelta y aL) (zip weights as)]

edgeDelta :: Label -- ^ Correct output Y for input X
          -> Activation -- ^ Predicted ouput for input X
          -> (Matrix Double, Matrix Double, Matrix Double)
edgeDelta y aL = ((aL - y) * sigmoid' aL, (1><1)[], (1><1)[])

delta :: (Matrix Double, Matrix Double, Matrix Double)
      -> (Weight, Activation)
      -> (Matrix Double, Matrix Double, Matrix Double)
delta (deltal, nW, nB) (wlp1, a) = ((tr wlp1 <> deltal) * sigmoid' a, deltal <> tr a, deltal)

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
    headW <- newWeight ls 784
    midW  <- replicateM (n-2) (newWeight ls ls)
    lastW <- newWeight 10 ls

    initB <- replicateM (n-1) (newBias ls 1)
    lastB <- newBias 10 1

    let weights = reverse $ headW : midW ++ [lastW]
    let biases  = reverse $ initB ++ [lastB]

    return net { weights = weights
               , biases  = biases }
  where
    newWeight = randn
    newBias r c = randn r c >>= return . foldl1 (|||) . replicate bs

randomMatrix :: Int -> Int -> IO (Matrix Double)
randomMatrix nrows ncols = do
    gaussList <- replicateM (ncols*nrows) gauss
    return $ (nrows><ncols) gaussList

loadData :: IO [(Image, Label)]
loadData = do
    trainImgs <- getData "train-images-idx3-ubyte.gz"
    trainLabels <- getData "train-labels-idx1-ubyte.gz"
    let labels = BL.toStrict trainLabels
    let imgs = BL.toStrict trainImgs
    let l = [vectorizeLabel $ getLabel n labels | n <- [0..49999]]
    let i = [getImage n imgs | n <- [0..49999]]
    return $ zip i l

loadTestData :: IO ([Image], [Int])
loadTestData = do
    imgs <- getData "t10k-images-idx3-ubyte.gz"
    labels <- getData "t10k-labels-idx1-ubyte.gz"
    let l = [fromIntegral $ BL.index labels (n+8) | n <- [0..9999]]
    let i = [getImage' n imgs | n <- [0..9999]]
    return (i, l)

getData :: FilePath -> IO BL.ByteString
getData path = do
    currentDir <- getCurrentDirectory
    fileData <- decompress <$> BL.readFile (currentDir ++ "/data/mnist_dataset/" ++ path)
    return fileData

getImage :: Int -> BS.ByteString -> Image
getImage n imgs = (784><1) [normalize $ BS.index imgs (16 + n*784 + s) | s <- [0..783]]

-- | `getImage` but lazy
getImage' :: Int64 -> BL.ByteString -> Image
getImage' n imgs = (784><1) [normalize $ BL.index imgs (16 + n*784 + s) | s <- [0..783]]

getLabel :: Num a => Int -> BS.ByteString -> a
getLabel n labels = fromIntegral $ BS.index labels (n+8)

getGuess :: Matrix Double -> Int
getGuess = fst . maxIndex

vectorizeLabel :: Int -> Matrix Double
vectorizeLabel l = (10><1) $ x ++ 1 : y
    where (x,y) = splitAt l [0,0..]

{-- HELPER FUNCTIONS --}
unpackTuples :: [(a, a)] -> [a]
unpackTuples []         = []
unpackTuples ((a,b):xs) = a:b:unpackTuples xs

packTuples :: [a] -> [(a, a)]
packTuples []       = []
packTuples (a:b:xs) = (a,b):packTuples xs

chunksOf :: Int -> [e] -> [[e]]
chunksOf i ls = map (take i) (build (splitter ls))
  where
    splitter :: [e] -> ([e] -> a -> a) -> a -> a
    splitter [] _ n = n
    splitter l c n  = l `c` splitter (drop i l) c n

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
