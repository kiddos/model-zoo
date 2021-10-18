export const graph = [
  {
    id: 'taipei-city',
    neighbor: ['new-taipei-city'],
  },
  {
    id: 'new-taipei-city',
    neighbor: ['taipei-city', 'keelung-city', 'taoyuan-city', 'yilan-county'],
  },
  {
    id: 'keelung-city',
    neighbor: ['new-taipei-city'],
  },
  {
    id: 'taoyuan-city',
    neighbor: ['new-taipei-city', 'yilan-county', 'hsinchu-county'],
  },
  {
    id: 'hsinchu-county',
    neighbor: ['taoyuan-city', 'yilan-county', 'hsinchu-city', 'miaoli-county', 'taichung-city'],
  },
  {
    id: 'hsinchu-city',
    neighbor: ['hsinchu-county', 'miaoli-county'],
  },
  {
    id: 'miaoli-county',
    neighbor: ['hsinchu-city', 'hsinchu-county', 'taichung-city'],
  },
  {
    id: 'taichung-city',
    neighbor: ['miaoli-county', 'hsinchu-county', 'yilan-county', 'changhua-county', 'nantou-county', 'hualien-county'],
  },
  {
    id: 'changhua-county',
    neighbor: ['taichung-city', 'nantou-county', 'yunlin-county'],
  },
  {
    id: 'nantou-county',
    neighbor: ['taichung-city', 'changhua-county', 'yunlin-county', 'chiayi-county', 'kaohsiung-city', 'hualien-county'],
  },
  {
    id: 'yunlin-county',
    neighbor: ['changhua-county', 'nantou-county', 'chiayi-county'],
  },
  {
    id: 'chiayi-county',
    neighbor: ['yunlin-county', 'nantou-county', 'kaohsiung-city', 'tainan-city', 'chiayi-city'],
  },
  {
    id: 'chiayi-city',
    neighbor: ['chiayi-county'],
  },
  {
    id: 'tainan-city',
    neighbor: ['chiayi-county', 'kaohsiung-city'],
  },
  {
    id: 'kaohsiung-city',
    neighbor: ['tainan-city', 'chiayi-county', 'nantou-county', 'hualien-county', 'taitung-county', 'pingtung-county'],
  },
  {
    id: 'pingtung-county',
    neighbor: ['kaohsiung-city', 'taitung-county'],
  },
  {
    id: 'taitung-county',
    neighbor: ['pingtung-county', 'kaohsiung-city', 'hualien-county'],
  },
  {
    id: 'hualien-county',
    neighbor: ['taitung-county', 'kaohsiung-city', 'nantou-county', 'taichung-city', 'yilan-county'],
  },
  {
    id: 'yilan-county',
    neighbor: ['new-taipei-city', 'taoyuan-city', 'hsinchu-county', 'taichung-city', 'hualien-county'],
  },
  {
    id: 'kinmen-county',
    neighbor: ['penghu-county', 'lienchiang-county'],
  },
  {
    id: 'penghu-county',
    neighbor: ['kinmen-county', 'lienchiang-county'],
  },
  {
    id: 'lienchiang-county',
    neighbor: ['penghu-county', 'kinmen-county'],
  }
];

const selected = ['#57AAF2', '#8DE879', '#F27164', '#F7AD55', '#B182EC', '#8CE6EA'];

export function welshPowell(graph) {
  const g = JSON.parse(JSON.stringify(graph)).sort((a, b) => b.neighbor.length - a.neighbor.length);
  const n = g.length;
  const colors = {};
  for (let color of selected) {
    for (let i = 0; i < n; ++i) {
      if (!colors[g[i].id]) {
        let found = false;
        for (let neighborId of g[i].neighbor) {
          if (colors[neighborId] === color) {
            found = true;
            break;
          }
        }

        if (!found) {
          colors[g[i].id] = color;
        }
      }
    } 
  }
  return colors;
}

export function getTaiwanColors() {
  return welshPowell(graph);
}
