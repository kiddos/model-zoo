import React from 'react';
import TaiwanMap from '@svg-maps/taiwan';
import classNames from 'classnames';
import Tooltip from '@material-ui/core/Tooltip';
import Switch from '@material-ui/core/Switch';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import { withStyles } from '@material-ui/core/styles';

import { toL, toC } from './coordinate';
import { getRainData } from './rain';
import { getTaiwanColors } from './coloring';
import './Taiwan.css';


const Location = props => {
  const { location, color } = props;
  const [hover, setHover] = React.useState(false);
  const { id, name, path } = location;

  return (
    <React.Fragment>
	    <path
	      id={id}
	      name={name}
	      d={path}
        fill={color ? color : '#898989'}
        stroke="#000000"
	      className={classNames('location', { hover })}
	      onMouseEnter={() => setHover(true)}
	      onMouseLeave={() => setHover(false)}>
	      <title>{name}</title>
	    </path>
	  </React.Fragment>
  )
}


const RecordTooltip = withStyles(theme => ({
  tooltip: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    fontSize: theme.typography.pxToRem(16),
    maxWidth: 600,
  }
}))(Tooltip);


const rows = [
  {key: 'locationName', name: '測站名稱'},
  {key: 'stationId', name: '測站ID'},
  {key: 'lat', name: 'latitude'},
  {key: 'lon', name: 'longitutde'},
  {key: 'CITY', name: '縣市'},
  {key: 'TOWN', name: '鄉鎮'},
  {key: 'obsTime', name: '觀測資料時間'},
  {key: 'MIN_10', name: '10分鐘累積雨量(mm)'},
  {key: 'RAIN', name: '60分鐘累積雨量(mm)'},
  {key: 'HOUR_3', name: '3小時累積雨量(mm)'},
  {key: 'HOUR_6', name: '6小時累積雨量(mm)'},
  {key: 'HOUR_12', name: '12小時累積雨量(mm)'},
  {key: 'HOUR_24', name: '24小時累積雨量(mm)'},
  {key: 'NOW', name: '本日累積雨量'},
  {key: 'latest_2days', name: '前1日0時到現在之累積雨量'},
  {key: 'latest_3days', name: '前2日0時到現在之累積雨量'},
]


const Record = props => {
  const { record, size = 5 } = props;
  const { lat, lon } = record;
  const {x, y} = toC(lat, lon);
  const ref = React.useRef();
  record['obsTime'] = record?.time?.obsTime;

  const content = (
    <div>
      <table>
        <tbody>
          {rows.map(({key, name}) => (
            <tr key={key}>
              <td>{name}</td>
              <td>{record[key]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <React.Fragment>
      <RecordTooltip title={content} placement="top">
        <circle
          ref={ref}
          cx={x}
          cy={y}
          r={size}
          fill="#BBBBBB"
          stroke="#000"/>
      </RecordTooltip>
    </React.Fragment>
  );
}


class Taiwan extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      x: 0,
      y: 0,
      records: [],
      debug: false,
      showRecords: true,
    }
  }

  componentDidMount() {
    getRainData().then(res => {
      let { success, records } = res;
      if (success) {
        records = records.location.map(l => {
          const r = {...l};
          l.parameter.forEach(p => r[p.parameterName] = p.parameterValue);
          l.weatherElement.forEach(w => r[w.elementName] = w.elementValue);
          delete r.parameter;
          delete r.weatherElement;
          return r;
        });
        const city = new Set();
        records.forEach(r => city.add(r.CITY));
        const town = new Set();
        records.forEach(r => town.add(r.TOWN));
        this.setState({ records, city, town });
      }
    });
  }

  handleMouse = e => {
    const x = e.clientX;
    const y = e.clientY;
    const ll = toL(x, y);
    const coord = toC(ll.lat, ll.lon);
    if (this.state.debug) {
      console.log({x, y}, ll, coord);
    }
    // this.setState({x, y});
  }

  handleDebug = () => {
    this.setState({ debug: !this.state.debug });
  }

  handleShowRecord = () => {
    this.setState({ showRecords: !this.state.showRecords });
  }

  render() {
    const { locations, label, viewBox } = TaiwanMap;
    const { records, city, town, debug, showRecords } = this.state;
    const colors = getTaiwanColors();

    return (
      <div className="map-container" onMouseMove={this.handleMouse}>
        <div className="map">
          <svg 
	          xmlns="http://www.w3.org/2000/svg"
	          viewBox={viewBox}
	          aria-label={label}>
	          {locations.map(location => <Location key={location.id} location={location} color={colors[location.id]}/>)}
	          {showRecords && records.slice(0, 1000).map((record, index) => (
	            <Record key={index} record={record}/>
            ))}
          </svg>
        </div>
        <div className="info">
          <h4>自動雨量站-雨量觀測資料</h4>
          <table>
            <tbody>
              <tr>
                <td>city:</td>
                <td>{city?.size}</td>
              </tr>
              <tr>
                <td>town:</td>
                <td>{town?.size}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div>
          <FormControlLabel
            control={
              <Switch
                checked={debug}
                variant="contained"
                color="primary"
                onChange={this.handleDebug}>debug</Switch>
            }
            label="debug"/>
          <FormControlLabel
            control={
              <Switch
                checked={showRecords}
                variant="contained"
                color="primary"
                onChange={this.handleShowRecord}>Show Records</Switch>
            }
            label="show records"/>
        </div>
      </div>
    );
  }
}

export default Taiwan;
