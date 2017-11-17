
'use strict';

module.exports = { // eslint-disable-line
  name: 'HI_djICA',
  version: '0.0.1',
  cwd: __dirname,
  local: {
    type: 'cmd',
    cmd: 'python',
    args: ['./djica_local.py'],
    verbose: true,
  },
  remote: {
    type: 'cmd',
    cmd: 'python',
    args: ['./djica_master.py'],
    verbose: true,
  },
  plugins:['group-step'],
};
