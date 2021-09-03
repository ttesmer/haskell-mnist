module Main where
import Lib
import qualified Data.Matrix as M

main :: IO ()
main = processMNIST

-- main :: IO ()
-- main = do
--     let x = M.fromLists [[0,0,1],
--                          [0,1,1],
--                          [1,0,1],
--                          [1,1,1]]
--     -- let y = M.colVector $ M.getCol 1 x
--     let y = M.fromList 4 1 [0,1,1,0]
-- 
--     syn0 <- randomMatrix 3 4
--     syn1 <- randomMatrix 4 1
-- 
--     -- trainBasic 10000 x y syn0
--     trainDeepNet 60000 x y syn0 syn1
