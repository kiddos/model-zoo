export const getRainData = () => {
  const url = 'https://opendata.cwb.gov.tw/api/v1/rest/datastore/O-A0002-001?Authorization=rdec-key-123-45678-011121314';
  return fetch(url).then(res => res.json());
}
