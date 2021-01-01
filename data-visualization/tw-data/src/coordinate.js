import { matrix, transpose, inv, multiply } from 'mathjs';

const computeT = () => {
  let X = transpose(matrix([
    [692, 1293, 1.0],
    [878, 325, 1.0],
    [386, 826, 1.0],
    [601, 40, 1.0],
    [63, 609, 1.0],
  ]));
  let Y = transpose(matrix([
    [21.897082, 120.858313, 1.0],
    [25.298443, 121.538076, 1.0],
    [23.553975, 119.694132, 1.0],
    [26.359486, 120.489795, 1.0],
    [24.427044, 118.459410, 1.0],
  ]));
  let Xt = transpose(X);
  const T = multiply(multiply(Y, Xt), inv(multiply(X, Xt)));
  return T;
}

const T = computeT();
const Tt = inv(T);

export const toL = (x, y) => {
  let X = transpose(matrix([[x, y, 1.0]]));
  let Y = multiply(T, X).toArray();
  return {lat: Y[0][0], lon: Y[1][0]};
}

export const toC = (lat, lon) => {
  let Y = transpose(matrix([[lat, lon, 1.0]]));
  let X = multiply(Tt, Y).toArray();
  return {x: X[0][0], y: X[1][0]};
}
