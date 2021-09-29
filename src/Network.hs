{-# LANGUAGE BangPatterns #-}
module Network
    ( train, 
      loadData,
      loadTestData,
      randomMatrix,
    ) where


import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString as BS
import qualified Data.Array.IO as A
import qualified Control.Monad as C
import qualified System.Random as R
import Numeric.LinearAlgebra hiding (normalize, build)
import System.Directory (getCurrentDirectory)
import Codec.Compression.GZip (decompress)
import Prelude hiding ((<>))
import GHC.Base (build)
import Data.List
 
createNetwork :: [Int] -> [Int]
createNetwork sizes = sizes

test :: (BS.ByteString, BS.ByteString) -- ^ Testing data
     -> [(Matrix Double, Matrix Double)] -- ^ Learned weights and biases
     -> Int -- ^ Index for image
     -> (Double, Double) -- ^ Keep track of how many test have been done
     -> Double -- ^ Number of tests that were correct.
     -> IO String
test (imgs,labels) learnedWaB i (n,nN) correct = do
    let image = getImageS i imgs 
    let label = getLabelS i labels
    let prediction = getGuess $ head $ feedforward image learnedWaB
    let updateCorrect = correct+(if (label == prediction) then 1 else 0)
    if i+1 == round nN
       then return $ (show (round updateCorrect)) ++ "/" ++ (show $ round nN) ++ " (" ++ (show $ ((updateCorrect/(nN-n))*100 :: Double)) ++  "%)"
       else test (imgs,labels) learnedWaB (i+1) (n-1,nN) (updateCorrect)
        where
            getImageS :: Int -> BS.ByteString -> Matrix Double
            getImageS n imgs = (784><1) [normalize . fromIntegral $ BS.index imgs (16 + n*784 + s) | s <- [0..783]]
            getLabelS :: Num a => Int -> BS.ByteString -> a
            getLabelS n labels = fromIntegral $ BS.index labels (n+8)
            renderPixel :: Double -> Char
            renderPixel n = let s = " .:oO@" in s !! floor (((n*255) * 6) / 256)

train :: Int -- ^ Number of epochs
      -> Int -- ^ Size of batches
      -> Double -- ^ Learning Rate
      -> [(Matrix Double, Matrix Double)] -- ^ Zipped weights and biases
      -> (BS.ByteString, BS.ByteString) -- ^ Training data
      -> (BS.ByteString, BS.ByteString) -- ^ Testing data
      -> IO [(Matrix Double, Matrix Double)]
train epochs batchSize eta weightsAndBiases trainingData testData = do
    indices <- shuffle [0..49999]
    learnedWaB <- updateBatches batchSize eta indices weightsAndBiases trainingData
    performance <- test testData learnedWaB 0 (10000, 10000) 0
    putStrLn $ "Epoch #" ++ (show $ 30-epochs) ++ ": " ++ performance
    if (epochs-1) == 0
      then return learnedWaB 
      else train (epochs-1) batchSize eta learnedWaB trainingData testData

updateBatches :: Int -- ^ Size of batch
              -> Double -- ^ Learning rate
              -> [Int] -- ^ Shuffled indices
              -> [(Matrix Double, Matrix Double)] -- ^ Zipped weights and biases
              -> (BS.ByteString, BS.ByteString) -- ^ Training data
              -> IO [(Matrix Double, Matrix Double)]
updateBatches batchSize eta indices weightsAndBiases trainingData = do
    (batch, newIndices) <- getNextBatch trainingData batchSize indices 
    let learnedWaB = updateWB eta batchSize weightsAndBiases $ trainBatch weightsAndBiases batch
    if (length newIndices) < batchSize
       then return learnedWaB
       else updateBatches batchSize eta newIndices learnedWaB trainingData


updateWB :: Double -- ^ Learning rate
         -> Int -- ^ Size of the batch
         -> [(Matrix Double, Matrix Double)] -- ^ List of zipped weight and bias matrices
         -> [Matrix Double] -- ^ List of zipped deltas; output from `trainBatch`
         -> [(Matrix Double, Matrix Double)]
updateWB eta batchSize weightsAndBiases nablaList = packTuples $ zipWith (-) (unpackTuples weightsAndBiases) (mapAvg nablaList) 
    where
        mapAvg :: [Matrix Double] -> [Matrix Double]
        mapAvg xs = (\nabla -> scale (eta/fromIntegral batchSize) nabla) <$> xs

trainBatch :: [(Matrix Double, Matrix Double)] -- ^ List of zipped weight and bias matrices
           -> [(Matrix Double, Matrix Double)] -- ^ List of zipped training examples (batch)
           -> [Matrix Double]
trainBatch wAb !((x1, y1):batch) = foldl' (\lastDeltas (x, y) -> zipWith (+) lastDeltas $ getDeltas x y) (getDeltas x1 y1) batch
    where 
        getDeltas :: Matrix Double -> Matrix Double -> [Matrix Double] 
        getDeltas x y = backprop y weights $ feedforward x wAb
            where weights = fst <$> wAb 

feedforward :: Matrix Double -- ^ Input X
            -> [(Matrix Double, Matrix Double)] -- ^ Zipped weights and biases
            -> [Matrix Double]
feedforward = scanr (\(w, b) a -> sigmoid $ w <> a + b)

backprop :: Matrix Double -- ^ Correct output Y
         -> [Matrix Double] -- ^ List of weight matrices
         -> [Matrix Double] -- ^ List of activation matrices
         -> [Matrix Double]
backprop y weights (aL:as)= unpackTuples $ drop 1 [(nw, nb) | (dn, nw, nb) <- scanl' delta (edgeDelta y aL) (zip weights as)]

edgeDelta :: Matrix Double -- ^ Correct output Y for input X
          -> Matrix Double -- ^ Predicted ouput for input X
          -> (Matrix Double, Matrix Double, Matrix Double)
edgeDelta y aL = ((aL - y) * sigmoid' aL, (1><1)[], (1><1)[])

delta :: (Matrix Double, Matrix Double, Matrix Double)
      -> (Matrix Double, Matrix Double) 
      -> (Matrix Double, Matrix Double, Matrix Double)
delta (deltal, nW, nB) (wlp1, a) = ((tr wlp1 <> deltal) * sigmoid' a, deltal <> tr a, deltal)

{-- Arithmetic Functions --}
cost :: Matrix Double -> Matrix Double -> Matrix Double
cost a y = ((a-y)**2) / 2

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + (exp (-x)))

sigmoid' :: Floating a => a -> a
sigmoid' x = x * (1 - x)

normalize :: Floating a => a -> a
normalize x = x / 255

gauss :: IO Double
gauss = do
    u1 <- R.randomRIO (0 :: Double, 1 :: Double)
    u2 <- R.randomRIO (0 :: Double, 1 :: Double)
    return $ (sqrt (-2*log u1)) * (cos (2*pi*u2))
    
{-- Data Processing Functions --}
randomMatrix :: Int -> Int -> IO (Matrix Double)
randomMatrix nrows ncols = do 
    gaussList <- C.replicateM (ncols*nrows) gauss
    return $ (nrows><ncols) gaussList

loadData :: IO (BS.ByteString, BS.ByteString)
loadData = do
    currentDir <- getCurrentDirectory
    trainImgs <- decompress <$> BL.readFile (currentDir ++ "/data/mnist_dataset/train-images-idx3-ubyte.gz")
    trainLabels <- decompress <$> BL.readFile (currentDir ++ "/data/mnist_dataset/train-labels-idx1-ubyte.gz")
    return (BL.toStrict trainImgs, BL.toStrict trainLabels)

loadTestData :: IO (BS.ByteString, BS.ByteString)
loadTestData = do
    currentDir <- getCurrentDirectory
    testImgs <- decompress <$> BL.readFile (currentDir ++ "/data/mnist_dataset/t10k-images-idx3-ubyte.gz")
    testLabels <- decompress <$> BL.readFile (currentDir ++ "/data/mnist_dataset/t10k-labels-idx1-ubyte.gz")
    return (BL.toStrict testImgs, BL.toStrict testLabels)

getNextBatch :: (BS.ByteString, BS.ByteString) 
             -> Int 
             -> [Int] 
             -> IO ([(Matrix Double, Matrix Double)], [Int])
getNextBatch (imgs, labels) size indices = do
    let (batchIndices, remainingIndices) = splitAt size indices
    let !batch = (\i -> getLabelAndImage i imgs labels) <$> batchIndices
    return (batch, remainingIndices) 

getLabelAndImage :: Int 
                 -> BS.ByteString 
                 -> BS.ByteString 
                 -> (Matrix Double, Matrix Double)
getLabelAndImage n imgs labels = (getImage n imgs, vectorizeLabel $ getLabel n labels)

getImage :: Int -> BS.ByteString -> Matrix Double
getImage n imgs = (784><1) [normalize . fromIntegral $ BS.index imgs (16 + n*784 + s) | s <- [0..783]]

getLabel :: Num a => Int -> BS.ByteString -> a
getLabel n labels = fromIntegral $ BS.index labels (n+8)

getGuess :: Matrix Double -> Int
getGuess = fst . maxIndex

vectorizeLabel :: Int -> Matrix Double
vectorizeLabel l = (10><1) $ x ++ 1 : y
    where (x,y) = splitAt l [0,0..]

{-- HELPER FUNCTIONS --}
unpackTuples :: [(a, a)] -> [a]
unpackTuples [] = []
unpackTuples ((a,b):xs) = a:b:unpackTuples xs

packTuples :: [a] -> [(a, a)] 
packTuples [] = []
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
        C.forM [1..n] $ \i -> do
            j <- R.randomRIO (i,n)
            vi <- A.readArray ar i
            vj <- A.readArray ar j
            A.writeArray ar j vi
            return vj
  where
    n = length xs
    newArray :: Int -> [a] -> IO (A.IOArray Int a)
    newArray n xs =  A.newListArray (1,n) xs
