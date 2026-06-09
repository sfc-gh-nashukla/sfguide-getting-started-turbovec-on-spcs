/*
 * TurboVec on SPCS - Cleanup
 * Removes all resources created by this quickstart
 */

-- Drop services
USE ROLE SYSADMIN;
DROP SERVICE IF EXISTS TURBOVEC_DEMO.PUBLIC.TURBOVEC;

-- Drop compute pools
DROP COMPUTE POOL IF EXISTS TURBOVEC_COMPUTE_POOL;
DROP COMPUTE POOL IF EXISTS JUPYTER_COMPUTE_POOL;

-- Drop database (includes stages, image repo, functions)
DROP DATABASE IF EXISTS TURBOVEC_DEMO;

-- Drop warehouse
DROP WAREHOUSE IF EXISTS TURBOVEC_WAREHOUSE;

-- Drop user and role
USE ROLE USERADMIN;
DROP USER IF EXISTS turbovec_user;

USE ROLE SECURITYADMIN;
DROP ROLE IF EXISTS TURBOVEC_ROLE;

-- Drop external access integration
USE ROLE ACCOUNTADMIN;
DROP INTEGRATION IF EXISTS turbovec_eai;
DROP NETWORK RULE IF EXISTS turbovec_egress_rule;
