import React from 'react';
import Head from 'next/head';
import Image from 'next/image';
import { Inter } from '@next/font/google';
import dynamic from 'next/dynamic';

import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';

const inter = Inter({ subsets: ['latin'] });

const DATASETS = [
  'Bladder',
  'Brain_Neurons',
  'Fat',
  'Kidney',
  'Lung',
  'Marrow',
  'Pancreas',
  'Spleen',
  'Tongue',
  'Brain_Microglia',
  'Colon',
  'Heart',
  'Liver',
  'Mammary',
  'Muscle',
  'Skin',
  'Thymus',
  'Trachea',
];

const VISUALIZATIONS = [
  {
    label: 'tSNE',
    value: 'tsne',
    axis: [
      {
        name: 'TSNE-1',
        output: 'x',
      },
      {
        name: 'TSNE-2',
        output: 'y',
      },
    ],
    classLabel: 'cell_ontology_class',
  },
  {
    label: 'UMAP',
    value: 'umap',
    axis: [
      {
        name: 'UMAP-1',
        output: 'x',
      },
      {
        name: 'UMAP-2',
        output: 'y',
      },
      {
        name: 'UMAP-3',
        output: 'z',
      },
    ],
    classLabel: 'cell_ontology_class',
  },
  {
    label: 'Louvain',
    value: 'louvain',
    axis: [
      {
        name: 'UMAP-1',
        output: 'x',
      },
      {
        name: 'UMAP-2',
        output: 'y',
      },
      {
        name: 'UMAP-3',
        output: 'z',
      },
    ],
    classLabel: 'louvain',
  },
  {
    label: 'Autoencoder PCA',
    value: 'ae_pca',
    axis: [
      {
        name: 'autoencoder_pca-1',
        output: 'x',
      },
      {
        name: 'autoencoder_pca-2',
        output: 'y',
      },
      {
        name: 'autoencoder_pca-3',
        output: 'z',
      },
    ],
    classLabel: 'cell_ontology_class',
  },
  {
    label: 'Autoencoder tSNE',
    value: 'ae_tsne',
    axis: [
      {
        name: 'autoencoder_tsne-1',
        output: 'x',
      },
      {
        name: 'autoencoder_tsne-2',
        output: 'y',
      },
      {
        name: 'autoencoder_tsne-3',
        output: 'z',
      },
    ],
    classLabel: 'cell_ontology_class',
  },
];

const Plot = dynamic(() => import('../components/Plot'), {
  suspense: true,
  ssr: false,
});

class Home extends React.PureComponent {
  state = {
    dataset: DATASETS[0],
    viz: VISUALIZATIONS[0].value,
  };

  handleSelectDataset = e => this.setState({ dataset: e.target.value });
  handleSelectViz = e => this.setState({ viz: e.target.value });

  render() {
    const { dataset, viz } = this.state;
    const { datasets } = this.props;
    const key = `${dataset}_${viz}`;
    const selectedDataset = datasets[key];
    const visualization = VISUALIZATIONS.find(v => v.value === viz);
    const { axis, classLabel } = visualization;
    return (
      <>
        <Head>Tabula Muris</Head>

        <h3 className={inter.className}>Tabula Muris</h3>

        <FormControl style={{ width: 200 }}>
          <InputLabel id="datasets">Datasets</InputLabel>
          <Select value={dataset} label="dataset" onChange={this.handleSelectDataset} size="small">
            {DATASETS.map(d => (
              <MenuItem key={d} value={d}>
                {d}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl style={{ marginLeft: 10, width: 200 }}>
          <InputLabel id="visualization">Visualization</InputLabel>
          <Select value={viz} label="visualization" onChange={this.handleSelectViz} size="small">
            {VISUALIZATIONS.map(({ label, value }) => (
              <MenuItem key={value} value={value}>
                {label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <div>
          <Plot
            dataset={selectedDataset}
            axis={axis}
            classLabel={classLabel}
            title={`${dataset}: ${visualization.label}`}
          />
        </div>
      </>
    );
  }
}

export default Home;

export async function getStaticProps() {
  const csvToJson = require('convert-csv-to-json');
  const datasets = {};
  for (let i = 0; i < DATASETS.length; ++i) {
    for (let j = 0; j < VISUALIZATIONS.length; ++j) {
      const csvFilePath = `../${DATASETS[i]}/${VISUALIZATIONS[j].value}.csv`;
      const json = csvToJson.fieldDelimiter(',').getJsonFromCsv(csvFilePath);
      const key = `${DATASETS[i]}_${VISUALIZATIONS[j].value}`;
      datasets[key] = json;
    }
  }

  // By returning { props: { posts } }, the Blog component
  // will receive `posts` as a prop at build time
  return {
    props: {
      datasets,
    },
  };
}
