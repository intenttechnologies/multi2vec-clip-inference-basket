'use strict'
const config = require('conventional-changelog-conventionalcommits');

module.exports = config({
    "issuePrefixes": ["PRO-"],
    "commitUrlFormat": "{{host}}/{{owner}}/{{repository}}/commit/{{hash}}",
    "issueUrlFormat": "https://linear.app/intenttech/issue/{{id}}/"
})