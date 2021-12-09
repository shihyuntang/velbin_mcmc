import numpy as np
from astropy.table import Table
import sys, os, argparse



def main(args):
    
    tb_org = Table.read('./Input/500pc_refined_memberlist_sub/{}.csv'.format(args.filename))
    
    rvuseN = 'Gaia_radial_velocity'
    rvuseNerr = 'er_Gaia_radial_velocity'
    tb_with_rv = tb_org[ (tb_org[rvuseN]>-9000) ]
    
    rv = np.array(tb_with_rv[rvuseN])
    erv   = np.array(tb_with_rv[rvuseNerr])
    
    mass     = tb_org['Mass']
    pmra     = np.array(tb_org['pmra'])
    epmra    = np.array(tb_org['er_pmra'])
    pmdec    = np.array(tb_org['pmdec'])
    epmdec   = np.array(tb_org['er_pmdec'])
    
    
    print("""\n
Target OC name (csv file name): {}
There are in total {} members
Only {} ({:1.1%}) of them have Gaia DR2 RVs
-rv [{:1.1f},{:1.1f},{:1.1f},{:1.1f},0.5] -pmra [{:1.1f},{:1.1f},{:1.1f},{:1.1f}] -pmdec [{:1.1f},{:1.1f},{:1.1f},{:1.1f}]
    """.format(args.filename, len(tb_org), len(tb_with_rv), len(tb_with_rv)/len(tb_org),
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