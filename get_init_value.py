import numpy as np
from astropy.table import Table
import sys, os, argparse



def main(args):
    
    tb_org = Table.read('./Input/500pc_refined_memberlist_sub/{}.csv'.format(args.filename))
    
    rvuseN = 'Gaia_radial_velocity'
    rvuseNerr = 'er_Gaia_radial_velocity'
    tb_with_rv = tb_with_rv[ (tb_with_rv[rvuseN]>-9000) ]
    
    rv = np.array(tb_with_rv[rvuseN])
    erv   = np.array(tb_with_rv[rvuseNerr])
    
    mass     = tb_with_rv['Mass']
    pmra     = np.array(tb_with_rv['pmra'])
    epmra    = np.array(tb_with_rv['er_pmra'])
    pmdec    = np.array(tb_with_rv['pmdec'])
    epmdec   = np.array(tb_with_rv['er_pmdec'])
    
    
    print("""\n
Target OC name (csv file name): {}
There are in total {} members
Only {} have Gaia DR2 RVs
-rv [{:1.1f},{:1.1f},{:1.1f},{:1.1f},0.5] -pmra [{:1.1f},{:1.1f},{:1.1f},{:1.1f}] -pmdec [{:1.1f},{:1.1f},{:1.1f},{:1.1f}]
    """.format(args.filename, len(tb_org), len(tb_with_rv),
               np.mean(rv),    np.std(rv),    np.mean(rv),    np.std(rv),
               np.mean(pmra),  np.std(pmra),  np.mean(pmra),  np.std(pmra),
               np.mean(pmdec), np.std(pmdec), np.mean(pmdec), np.std(pmdec),
               )
    )
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                                     prog        = 'modified the velbin code',
                                     description = '''
                                     modified velbin code.
                                     ''',
                                     epilog = "Contact author: sytang@lowell.edu")
    parser.add_argument("filename",                          action="store",
                        help="Enter your filename you wish you use under ./Input/", type=str)

    # global args
    args = parser.parse_args()
    main(args)