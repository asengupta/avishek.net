<?php
	if (!class_exists('WikiTagSet', false)) {
		class WikiTagSet extends TagSet {
		
			function getTagSetName() {
				return 'MediaWiki';
			}
		
			function getExternalPluginInfos() {
				return array(
					'name' => 'MediaWiki Markup for WordPress',
					'website' => 'http://zechs.dyndns.org/wordpress/?page_id=126'
				);
			}
		
			function isExternalPluginActive() {
				return function_exists('wpwiki');
			}
		
			function isExternalPluginInstalled() {
				return is_file($this->getPluginDir().'wp-mediawiki.php');
			}
		}
	}
	$tagset = new WikiTagSet();
?>