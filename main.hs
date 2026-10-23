module Main where

-- Soma de dois números
soma :: Int -> Int -> Int
soma a b = a + b

-- Fatorial
fatorial :: Integer -> Integer
fatorial 0 = 1
fatorial n = n * fatorial (n - 1)

-- Lista de quadrados
quadrados :: [Int] -> [Int]
quadrados xs = map (^ 2) xs

main :: IO ()
main = do
  putStrLn "=== Programa em Haskell ==="
  print (soma 3 4)
  print (fatorial 5)
  print (quadrados [1, 2, 3, 4, 5])
