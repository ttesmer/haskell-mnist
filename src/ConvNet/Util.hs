module ConvNet.Util where

import           Codec.Compression.GZip    (decompress)
import           Control.DeepSeq
import           Control.Monad             as C
import qualified Data.Array.IO             as Arr
import qualified Data.ByteString           as BS
import qualified Data.ByteString.Lazy      as BL
import           Data.List                 as L
import           Data.Massiv.Array         as A
import           Data.Massiv.Array.Numeric as A
import           GHC.Base                  (build)
import           Prelude                   as P hiding ((<>))
import           System.Directory          (getCurrentDirectory)
import qualified System.Random             as R

{-- Arithmetic Functions --}
loadData :: IO ()
loadData = do
    trainImgs <- return . BL.toStrict =<< getData "train-images-idx3-ubyte.gz"
    trainLabels <- return . BL.toStrict =<< getData "train-labels-idx1-ubyte.gz"
    let imgs_ :: Array U Ix3 Double
        imgs_ = A.fromLists' Par $
            (getImage trainImgs) <$> [0..49999]
        {-# INLINE imgs_ #-}
        labels_ :: Array U Ix3 Double
    return ()

{- 
loadTestData :: IO [(Matrix Double, Int)]
loadTestData = do
    testImgs <- getData "t10k-images-idx3-ubyte.gz"
    testLabels <- getData "t10k-labels-idx1-ubyte.gz"
    let labels = BL.toStrict testLabels
    let imgs = BL.toStrict testImgs
    return [(asColumn $ getImage n imgs, getLabel n labels) | n <- [0..9999]]
-}

getData :: FilePath -> IO BL.ByteString
getData path = do
    currentDir <- getCurrentDirectory
    decompress
        <$> BL.readFile (currentDir ++ "/data/mnist_dataset/" ++ path)

getImage :: BS.ByteString -> Int -> [[Double]]
getImage imgs n = chunkList 28 $
    [normalize $
        BS.index imgs (16 + n*784 + s) |
            s <- [0..783]]

getLabel :: Num a => BS.ByteString -> Int -> a
getLabel labels n = fromIntegral $ BS.index labels (n+8)

vectorizeLabel :: Int -> Vector DL Double
vectorizeLabel l = makeVectorR DL Par 10 (\i -> if i /= l then 0 else 1)

sumRows :: Array U Ix2 Double -> Vector D Double
sumRows = sumArrays' . innerSlices

normalize :: (Integral a, Floating b) => a -> b
normalize x = fromIntegral x / 255

-- | Chunk list into chunks of size n
chunkList :: Int -> [a] -> [[a]]
chunkList n xs = P.takeWhile (not.null) $ L.unfoldr (Just . splitAt n) xs

-- | Randomly shuffle a list
--   /O(N)/
shuffle :: [a] -> IO [a]
shuffle xs = do
        ar <- newArray n xs
        C.forM [1..n] $ \i -> do
            j <- R.randomRIO (i,n)
            vi <- Arr.readArray ar i
            vj <- Arr.readArray ar j
            Arr.writeArray ar j vi
            return vj
  where
    n = length xs
    newArray :: Int -> [a] -> IO (Arr.IOArray Int a)
    newArray n xs = Arr.newListArray (1,n) xs

-- | Randomly shuffle the 2D arrays
--  contained within a 3D array.
--  Has to be `compute`d to a
--  Manifest representation.
--
-- ==== __Examples__
--
-- >>> import Data.Massiv.Array as A
-- >>> :load Helper.hs
-- >>> arr = makeArrayLinear Par (Sz3 2 3 4) fromIntegral :: Array U Ix3 Double
-- >>> arr
-- Array U Par (Sz (2 :> 3 :. 4))
--  [ [ [ 0.0, 1.0, 2.0, 3.0 ]
--    , [ 4.0, 5.0, 6.0, 7.0 ]
--    , [ 8.0, 9.0, 10.0, 11.0 ]
--    ]
--  , [ [ 12.0, 13.0, 14.0, 15.0 ]
--    , [ 16.0, 17.0, 18.0, 19.0 ]
--    , [ 20.0, 21.0, 22.0, 23.0 ]
--    ]
--  ]
-- >>> shuffle3d arr
-- Array DL Par (Sz (2 :> 3 :. 4))
--  [ [ [ 12.0, 13.0, 14.0, 15.0 ]
--    , [ 16.0, 17.0, 18.0, 19.0 ]
--    , [ 20.0, 21.0, 22.0, 23.0 ]
--    ]
--  , [ [ 0.0, 1.0, 2.0, 3.0 ]
--    , [ 4.0, 5.0, 6.0, 7.0 ]
--    , [ 8.0, 9.0, 10.0, 11.0 ]
--    ]
--  ]
shuffle3d :: Array U Ix3 Double -> IO (Array U Ix3 Double)
shuffle3d arr = shuffle (A.toLists3 arr) >>= fromListsM Par
