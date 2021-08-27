module Main where
import Lib
import qualified System.Random as R
import qualified Control.Monad as CM
import qualified Data.Matrix as M

main :: IO ()
main = do

    let x = M.fromLists [[1,0,1],[1,1,1],[0,0,1]]
    let y = M.transpose $ M.fromLists [[1,1,0]]

    putStrLn "Base Array (X):"
    print x
    putStrLn "Goal Array (Y):"
    print y

    syn0 <- (CM.replicateM 3 $ (R.randomRIO (0 :: Double, 1 :: Double))) >>= (\x -> return $ M.fromList 3 1 ((\y -> 2*y -1) <$> x))

    putStrLn "Result After Training:"
    train 100000 x y syn0
