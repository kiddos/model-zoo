import Plot from 'react-plotly.js';

const COLORS = [
  '#AF5FFF',
  '#5FAAFF',
  '#7CD582',
  '#E9A95C',
  '#F35D5D',
  '#7DE2AF',
  '#E4E723',
  '#5E46D1',
];

export default function PlotComponent({ dataset, axis, classLabel, title }) {
  const uniqueClasses = dataset
    ?.map(row => row[classLabel])
    .filter((row, index, a) => a.indexOf(row) === index);
  const data = uniqueClasses?.map((c, index) => {
    const type = axis.length === 2 ? 'scatter' : 'scatter3d';
    // const color = '#' + Math.floor(Math.random() * 16777215).toString(16);
    const color = COLORS[index % COLORS.length];
    const scatter = { name: c, text: c, type, mode: 'markers', opacity: 0.66, marker: { color } };
    axis.forEach(({ name, output }) => {
      scatter[output] = dataset
        .filter(row => row[classLabel] === c)
        .map(row => parseFloat(row[name]));
    });
    return scatter;
  });
  return <Plot data={data} layout={{ width: 1100, height: 690, title }} />;
}
