module Helper where

import           Codec.Compression.GZip (decompress)
import           Control.DeepSeq
import           Control.Monad
import qualified Data.Array.IO          as A
import qualified Data.ByteString        as BS
import qualified Data.ByteString.Lazy   as BL
import           Data.List
import           GHC.Base               (build)
import           Numeric.LinearAlgebra  hiding (build, normalize)
import           Prelude                hiding ((<>))
import           System.Directory       (getCurrentDirectory)
import qualified System.Random          as R

type Image        = Vector Double
type Label        = Vector Double

{-- Arithmetic Functions --}
normalize :: (Integral a, Floating b) => a -> b
normalize x = fromIntegral x / 255

divBy :: Double -> Matrix Double -> Matrix Double
divBy x = scale (1/x)

loadData :: IO ([(Image, Label)], [(Matrix Double, Int)])
loadData = do
    trainImgs <- getData "train-images-idx3-ubyte.gz"
    trainLabels <- getData "train-labels-idx1-ubyte.gz"
    let labels = BL.toStrict trainLabels
    let imgs = BL.toStrict trainImgs
    let trainData = force [(getImage n imgs, vectorizeLabel $ getLabel n labels) | n <- [0..49999]]
    let validationData = force [(asColumn $ getImage n imgs, getLabel n labels) | n <- [50000..59999]]
    return (trainData, validationData)

loadTestData :: IO [(Matrix Double, Int)]
loadTestData = do
    testImgs <- getData "t10k-images-idx3-ubyte.gz"
    testLabels <- getData "t10k-labels-idx1-ubyte.gz"
    let labels = BL.toStrict testLabels
    let imgs = BL.toStrict testImgs
    return [(asColumn $ getImage n imgs, getLabel n labels) | n <- [0..9999]]

getData :: FilePath -> IO BL.ByteString
getData path = do
    currentDir <- getCurrentDirectory
    decompress
        <$> BL.readFile (currentDir ++ "/data/mnist_dataset/" ++ path)

getImage :: Int -> BS.ByteString -> Image
getImage n imgs = fromList [normalize $ BS.index imgs (16 + n*784 + s) | s <- [0..783]]

squareImgMats :: [(Image, Label)] -> [(Matrix Double, Label)]
squareImgMats zippedData = zip (map (reshape 28) imgs) labels
    where (imgs, labels) = unzip zippedData

getLabel :: Num a => Int -> BS.ByteString -> a
getLabel n labels = fromIntegral $ BS.index labels (n+8)

vectorizeLabel :: Int -> Vector Double
vectorizeLabel l = fromList $ x ++ 1 : y
    where (x,y) = splitAt l $ replicate 9 0

toMatrix :: Vector Double -> Matrix Double
toMatrix = reshape 1

sumRows :: (Num a, Element a) => Matrix a -> Vector a
sumRows m = fromList $ sum <$> toLists m

-- | Chunk list into chunks of size n
chunkList :: Int -> [a] -> [[a]]
chunkList n xs = takeWhile (not.null) $ unfoldr (Just . splitAt n) xs

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
